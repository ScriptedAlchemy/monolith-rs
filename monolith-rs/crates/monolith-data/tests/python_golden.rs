use monolith_proto::idl::matrix::proto::LineId;
use monolith_proto::monolith::io::proto::feature;
use monolith_proto::Example;
use prost::Message;

#[test]
fn decode_python_generated_example_bytes() {
    let bytes = include_bytes!("fixtures/example_python.bin");
    let ex = Example::decode(&bytes[..]).expect("decode Example");

    // Feature
    assert_eq!(ex.named_feature.len(), 1);
    let nf = &ex.named_feature[0];
    assert_eq!(nf.name, "user_id");
    let feat = nf.feature.as_ref().expect("feature");
    match &feat.r#type {
        Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![12345u64]),
        other => panic!("unexpected feature type: {:?}", other),
    }

    // Label
    assert_eq!(ex.label, vec![1.0]);

    // LineId
    let lid: &LineId = ex.line_id.as_ref().expect("line_id");
    assert_eq!(lid.uid, Some(42));
    assert_eq!(lid.req_time, Some(1700000000));
}
