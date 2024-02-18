use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_l2_bias() -> Tensor<FP16x16> {
    let mut shape = array![10];

    let mut data = array![FP16x16 { mag: 4297, sign: true }, FP16x16 { mag: 21430, sign: false }, FP16x16 { mag: 8160, sign: true }, FP16x16 { mag: 3377, sign: false }, FP16x16 { mag: 34081, sign: true }, FP16x16 { mag: 27693, sign: false }, FP16x16 { mag: 34291, sign: true }, FP16x16 { mag: 19238, sign: false }, FP16x16 { mag: 19336, sign: true }, FP16x16 { mag: 8193, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}