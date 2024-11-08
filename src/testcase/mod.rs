use std::{alloc::{alloc, Layout}, ffi::CStr, ptr};

use crate::{nvrtc::Nvrtc, test::TestCase};

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

static PTX_HEADER: &str = "
.version 6.5
.target sm_30
.address_size 64
";

/// Prepare test sources to be loaded as a module on the device.
/// NOTE: This assumes all arguments are u64.
fn prepare_test_module(args: &[&str], body: &str, nvrtc: &Option<Nvrtc>) -> String {
    /// Turn argument names into PTX test function signature.
    fn fmt_ptx_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!(".param .u64 {}", a)).collect();
        format!(".entry run({})", args.join(", "))
    }

    /// Turn argument names into CUDA test function signature.
    fn fmt_cuda_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!("uint64_t * {}", a)).collect();
        format!("__global__ void run({})", args.join(", "))
    }

    /// Turn argument names into CUDA inline PTX parameter list.
    fn fmt_cuda_inline_ptx_params(args: &[&str]) -> String {
        args.iter().map(|a| format!(r#""l"({})"#, a)).collect::<Vec<_>>().join(", ")
    }

    /// Transform raw PTX into CUDA inline PTX function body.
    fn ptx_to_inline(args: &[&str], body: &str) -> String {
        let mut body = body.to_string();

        // Substitute mentions of function arguments in raw PTX with inline PTX parameters
        for (arg_index, arg_name) in args.iter().enumerate() {
            // NOTE: This assumes the arguments are only used inside of `[]` (e.g. for `ld.param.u64`).
            //       Otherwise this needs to be replaced with a proper regex that matches whole words.
            let pattern = format!("[{}]", arg_name);

            body = body.replace(&pattern, &format!("[%{}]", arg_index));
        }

        body = body.lines().map(|l| format!("\"{}\"\n", l)).collect::<Vec<_>>().join("    ");

        format!(
            "asm({}    :: {});",
            body,
            fmt_cuda_inline_ptx_params(args),
        )
    }

    if let Some(nvrtc) = nvrtc {
        let source_cuda = format!(
            "{} {{\n{}\n}}\0",
            fmt_cuda_signature(args),
            ptx_to_inline(args, body),
        );

        let mut program = ptr::null_mut();
        unsafe { nvrtc.nvrtcCreateProgram(&mut program, source_cuda.as_ptr() as _, ptr::null(), 0, ptr::null(), ptr::null()) }.unwrap();
        unsafe { nvrtc.nvrtcCompileProgram(program, 0, ptr::null()) }.unwrap();

        let mut ptx_size = 0;
        unsafe { nvrtc.nvrtcGetPTXSize(program, &mut ptx_size) }.unwrap();

        println!("PTX size: {}", ptx_size);

        let layout = Layout::array::<core::ffi::c_char>(ptx_size).unwrap();
        let source_ptx_buffer = unsafe { alloc(layout) };

        unsafe { nvrtc.nvrtcGetPTX(program, source_ptx_buffer as _ ) }.unwrap();

        let source_ptx_cstr = unsafe { CStr::from_ptr(source_ptx_buffer as _) };
        let source_ptx = String::from_utf8_lossy(source_ptx_cstr.to_bytes()).to_string();

        println!("{}", source_ptx);

        unsafe { nvrtc.nvrtcDestroyProgram(&mut program) }.unwrap();

        source_ptx
    } else {
        format!(
            "{}\n{}\n{{\n{}\n}}\0",
            PTX_HEADER,
            fmt_ptx_signature(args),
            body,
        )
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
