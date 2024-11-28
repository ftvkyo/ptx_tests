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
    pub verbose: bool,

    pub cuda: Cuda,
    pub nvrtc: Option<Nvrtc>,
}

// TODO: split into a trait with 2 implementations
impl TestContext {
    /// Turn argument names into PTX test function signature.
    fn fmt_ptx_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!(".param .u64 {}", a)).collect();
        format!(".entry run({})", args.join(", "))
    }

    fn fmt_ptx_params_load(args: &[&str]) -> String {
        let mut text = String::new();
        for arg in args {
            text.push_str(&format!(".reg .u64    {name}_addr;\n", name = arg));
            text.push_str(&format!("ld.param.u64 {name}_addr, [{name}];\n", name = arg));
        }
        text
    }

    /// Turn argument names into CUDA test function signature.
    fn fmt_cuda_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!("uint64_t * {}", a)).collect();
        format!("extern \"C\" __global__ void run({})", args.join(", "))
    }

    fn fmt_cuda_inline_ptx_params_load(args: &[&str]) -> String {
        let mut text = String::new();
        for (arg_index, arg_name) in args.iter().enumerate() {
            text.push_str(&format!(".reg .u64 {name}_addr;\n", name = arg_name));
            text.push_str(&format!("mov.u64   {name}_addr, %{index};\n", name = arg_name, index = arg_index));
        }
        text
    }

    /// Turn argument names into CUDA inline PTX parameter list.
    fn fmt_cuda_inline_ptx_params(args: &[&str]) -> String {
        args.iter().map(|a| format!(r#""l"({})"#, a)).collect::<Vec<_>>().join(", ")
    }

    /// Transform raw PTX into CUDA inline PTX function body.
    fn ptx_to_inline(args: &[&str], body: &str) -> String {
        let mut body = body.to_string();

        // Escape "%" (used for things like %tid (thread id) etc.)
        body = body.replace("%", "%%");

        body = format!(
            "{}\n{}",
            Self::fmt_cuda_inline_ptx_params_load(args),
            body,
        );

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

            if self.verbose {
                for (i, line) in source_cuda.lines().enumerate() {
                    println!("{:3} |  {}", i + 1, line);
                }
            }

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
                "{}\n{}\n{{\n{}\n{}\nret;\n}}\0",
                header,
                Self::fmt_ptx_signature(args),
                Self::fmt_ptx_params_load(args),
                body,
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
