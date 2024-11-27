#![allow(internal_features)]
#![feature(link_llvm_intrinsics)]
#![feature(f16)]
#![feature(c_size_t)]

use std::ptr;

use bpaf::Bpaf;
use regex::{self, Regex};

use cuda::Cuda;
use test::TestError;
use testcase::*;

mod common;
mod cuda;
mod nvrtc;
mod test;
mod testcase;

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options)]
pub enum Arguments {
    List {
        /// list all available tests, execute no tests
        #[bpaf(short, long)]
        list: (),
    },
    Run {
        /// only tests matching this regex will be executed
        #[bpaf(short, long)]
        filter: Option<String>,

        /// path to CUDA shared library under testing, for example C:\Windows\System32\nvcuda.dll or /usr/lib/x86_64-linux-gnu/libcuda.so
        #[bpaf(positional("cuda"))]
        cuda: String,
    },
}

fn main() {
    let args = arguments().run();
    std::process::exit(run(args));
}

fn run(args: Arguments) -> i32 {
    let mut failures = 0;
    let mut tests = tests();
    tests.sort_unstable_by_key(|t| t.name.clone());
    match args {
        Arguments::List { .. } => {
            for test in tests {
                println!("{}", test.name);
            }
        }
        Arguments::Run { filter, cuda } => {
            if let Some(filter) = filter {
                let re = Regex::new(&filter).unwrap();
                tests = tests.into_iter().filter(|t| re.is_match(&t.name)).collect();
            }
            let cuda = Cuda::new(cuda);
            unsafe { cuda.cuInit(0) }.unwrap();
            let mut ctx = ptr::null_mut();
            unsafe { cuda.cuCtxCreate_v2(&mut ctx, 0, 0) }.unwrap();
            for t in tests {
                match (t.test)(&cuda) {
                    Ok(()) => println!("{}: OK", t.name),
                    Err(TestError::Mismatch(e)) => {
                        println!(
                            "{}: FAIL: Input {}, computed on GPU {}, computed on CPU {}",
                            t.name, e.input, e.output, e.expected
                        );
                        failures += 1;
                    }
                    Err(TestError::Miscompile(name)) => {
                        println!("{}: FAIL: Compilation mismatch", name);
                        failures += 1;
                    }
                }
            }
        }
    }
    failures
}

#[macro_export]
macro_rules! impl_library {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* $(,)* ) -> $ret_type:path);* $(;)*) => {
        $(
            #[allow(non_snake_case)]
            #[allow(improper_ctypes)]
            pub unsafe fn $fn_name(&self,  $( $arg_id : $arg_type),*) -> $ret_type {
                let fn_: libloading::Symbol<unsafe extern $abi fn( $($arg_type),*) -> $ret_type> =
                    self.library.get(concat!(stringify!($fn_name), "\0").as_bytes()).unwrap();
                fn_( $($arg_id),*)
            }
        )*
    };
}
