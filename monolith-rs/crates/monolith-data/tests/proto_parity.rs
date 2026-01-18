use monolith_data::example::{add_feature, create_example};
use monolith_proto::monolith::io::proto::feature;
use monolith_proto::Example;
use prost::Message;

#[test]
fn example_encode_decode_roundtrip() {
    let mut ex = create_example();
    add_feature(&mut ex, "user_id", vec![12345], vec![1.0]);

    let bytes = ex.encode_to_vec();
    let decoded = Example::decode(bytes.as_slice()).unwrap();
    assert_eq!(decoded.named_feature.len(), 1);

    let nf = &decoded.named_feature[0];
    assert_eq!(nf.name, "user_id");
    let feat = nf.feature.as_ref().unwrap();
    match &feat.r#type {
        Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![12345]),
        other => panic!("expected FidV2List, got {:?}", other),
    }
}
