use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_l1_bias() -> Tensor<FP16x16> {
    let mut shape = array![10];

    let mut data = array![FP16x16 { mag: 7814, sign: false }, FP16x16 { mag: 17987, sign: true }, FP16x16 { mag: 31823, sign: false }, FP16x16 { mag: 13775, sign: true }, FP16x16 { mag: 47083, sign: false }, FP16x16 { mag: 23125, sign: false }, FP16x16 { mag: 28295, sign: false }, FP16x16 { mag: 38347, sign: false }, FP16x16 { mag: 20452, sign: true }, FP16x16 { mag: 27263, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}