use std::{alloc::{alloc, Layout}, ffi::CStr, ptr};

use crate::{cuda::Cuda, nvrtc::Nvrtc, test::{TestCase, TestCommon}};

mod bfe;
mod bfi;
mod brev;
mod cos;
mod cvt;
mod lg2;
mod minmax;
mod rcp;
mod rsqrt;
mod shift;
mod sin;
mod sqrt;

pub struct TestContext {
    pub cuda: Cuda,
    pub nvrtc: Option<Nvrtc>,
}

impl TestContext {
    /// Turn argument names into PTX test function signature.
    fn fmt_ptx_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!(".param .u64 {}", a)).collect();
        format!(".entry run({})", args.join(", "))
    }

    /// Turn argument names into CUDA test function signature.
    fn fmt_cuda_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!("uint64_t * {}", a)).collect();
        format!("extern \"C\" __global__ void run({})", args.join(", "))
    }

    /// Turn argument names into CUDA inline PTX parameter list.
    fn fmt_cuda_inline_ptx_params(args: &[&str]) -> String {
        args.iter().map(|a| format!(r#""l"({})"#, a)).collect::<Vec<_>>().join(", ")
    }

    fn ptx_to_standalone(body: &str) -> String {
        let mut body = body.to_string();

        // In standalone PTX, we need to load the actual values of arguments
        body = body.replace("<LOAD_ARG>", "ld.param.u64");

        body
    }

    /// Transform raw PTX into CUDA inline PTX function body.
    fn ptx_to_inline(args: &[&str], body: &str) -> String {
        let mut body = body.to_string();

        // In inline PTX, the `asm(...)` block does `ld` for us
        body = body.replace("<LOAD_ARG>", "mov.u64");

        // Escape "%" (used for things like %tid (thread id) etc.)
        body = body.replace("%", "%%");

        // Substitute mentions of function arguments in raw PTX with inline PTX parameters
        for (arg_index, arg_name) in args.iter().enumerate() {
            // NOTE: This assumes the arguments are only used as source address operands (`[argument]`).
            //       Otherwise this needs to be replaced with a proper regex that matches whole words.
            let pattern = format!("[{}]", arg_name);

            // NOTE: This removes the square brackets because we use `mov` instead of `ld.param` in inline PTX.
            body = body.replace(&pattern, &format!("%{}", arg_index));
        }

        body = body.lines().map(|l| format!("\"{}\"\n", l)).collect::<Vec<_>>().join("    ");

        format!(
            "asm({}    :: {});",
            body,
            Self::fmt_cuda_inline_ptx_params(args),
        )
    }

    /// Prepare test sources to be loaded as a module on the device.
    /// NOTE: This assumes all arguments are u64.
    pub fn prepare_test_source<T: TestCommon>(&self, t: &T) -> String {
        let body = t.ptx();
        let args = t.ptx_args();
        let header = t.ptx_header();

        if let Some(nvrtc) = &self.nvrtc {
            let source_cuda = format!(
                "{} {{\n{}\n}}\0",
                Self::fmt_cuda_signature(args),
                Self::ptx_to_inline(args, &body),
            );

            let mut program = ptr::null_mut();
            unsafe { nvrtc.nvrtcCreateProgram(&mut program, source_cuda.as_ptr() as _, ptr::null(), 0, ptr::null(), ptr::null()) }.unwrap();
            unsafe { nvrtc.nvrtcCompileProgram(program, 0, ptr::null()) }.unwrap();

            let mut ptx_size = 0;
            unsafe { nvrtc.nvrtcGetPTXSize(program, &mut ptx_size) }.unwrap();

            let layout = Layout::array::<core::ffi::c_char>(ptx_size).unwrap();
            let source_ptx_buffer = unsafe { alloc(layout) };

            unsafe { nvrtc.nvrtcGetPTX(program, source_ptx_buffer as _ ) }.unwrap();

            let source_ptx_cstr = unsafe { CStr::from_ptr(source_ptx_buffer as _) };
            let source_ptx = String::from_utf8_lossy(source_ptx_cstr.to_bytes()).to_string();

            unsafe { nvrtc.nvrtcDestroyProgram(&mut program) }.unwrap();

            source_ptx
        } else {
            format!(
                "{}\n{}\n{{\n{}\nret;\n}}\0",
                header,
                Self::fmt_ptx_signature(args),
                Self::ptx_to_standalone(&body),
            )
        }
    }
}

pub fn tests() -> Vec<TestCase> {
    let mut tests = vec![
        bfe::rng_u32(),
        bfe::rng_s32(),
        bfe::rng_u64(),
        bfe::rng_s64(),
        bfi::rng_b32(),
        bfi::rng_b64(),
        brev::b32(),
    ];
    tests.extend(cvt::all_tests());
    tests.extend(rcp::all_tests());
    tests.extend(shift::all_tests());
    tests.extend(minmax::all_tests());
    tests.extend(sqrt::all_tests());
    tests.extend(rsqrt::all_tests());
    tests.extend(sin::all_tests());
    tests.extend(cos::all_tests());
    tests.extend(lg2::all_tests());
    tests
}
