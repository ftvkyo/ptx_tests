use crate::common::{self, flush_to_zero_f32};
use crate::cuda::Cuda;
use crate::test::{self, RangeTest, TestCase, TestCommon};
use core::f32;
use std::mem;

pub static PTX: &str = include_str!("sin.ptx");

pub(crate) fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(sin(ftz));
    }
    tests
}

fn sin(ftz: bool) -> TestCase {
    let test = Box::new(move |cuda: &Cuda| test::run_range::<Sin>(cuda, Sin { ftz }));
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("sin_approx{}", ftz), test)
}

pub struct Sin {
    ftz: bool,
}

const APPROX_TOLERANCE: f64 = 0.00000051106141211332948885584179164092160363501768f64; // 2^-20.9

impl TestCommon for Sin {
    type Input = f32;

    type Output = f32;

    fn ptx(&self) -> String {
        let ftz = if self.ftz { ".ftz" } else { "" };
        let mut src = PTX.replace("<FTZ>", &ftz);
        src.push('\0');
        src
    }

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn sin_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.is_subnormal() && f.is_sign_negative() => -0.0,
                f if f.to_ne_bytes() == (-0.0f32).to_ne_bytes() => -0.0,
                0.0 => 0.0,
                f if f.is_subnormal() && f.is_sign_positive() => 0.0,
                f32::INFINITY => f32::NAN,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if let Some(mut expected) = sin_approx_special(input) {
            flush_to_zero_f32(&mut expected, self.ftz);
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let precise_result = sin_host(input);
            let mut result_f32 = precise_result as f32;
            flush_to_zero_f32(&mut result_f32, self.ftz);
            let precise_output = output as f64;
            let diff = (precise_output - result_f32 as f64).abs();
            if diff <= APPROX_TOLERANCE {
                Ok(())
            } else {
                Err(precise_result as f32)
            }
        }
    }
}

const RANGE_MIN: f32 = 0f32;
const RANGE_MAX: f32 = f32::consts::FRAC_PI_2;

impl RangeTest for Sin {
    const MAX_VALUE: u32 =
        (unsafe { mem::transmute::<_, u32>(RANGE_MAX) - mem::transmute::<_, u32>(RANGE_MIN) }) + 36;

    fn generate(&self, input: u32) -> Self::Input {
        let max_number = unsafe { mem::transmute::<_, u32>(RANGE_MAX) };
        if input > max_number {
            match input - max_number {
                1 => f32::NEG_INFINITY,
                2 => common::MAX_NEGATIVE_SUBNORMAL,
                3 => -0.0,
                4 => 0.0,
                5 => common::MAX_POSITIVE_SUBNORMAL,
                6 => f32::INFINITY,
                7 => f32::NAN,
                _ => 0.0,
            }
        } else {
            unsafe { mem::transmute::<_, f32>(input + mem::transmute::<_, u32>(RANGE_MIN)) }
        }
    }
}

fn sin_host(input: f32) -> f64 {
    let input = input as f64;
    input.sin()
}