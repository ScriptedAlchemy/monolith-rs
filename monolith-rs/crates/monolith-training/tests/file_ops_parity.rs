use monolith_data::TFRecordReader;
use monolith_proto::monolith::hash_table::EntryDump;
use monolith_training::WritableFile;
use prost::Message;
use std::fs::File;
use std::io::{BufReader, ErrorKind};

#[test]
fn test_writable_file_append_entry_dump_tfrecord_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("entry_dump.tfrecord");

    let f = WritableFile::new(&path).unwrap();
    f.append_entry_dump(
        &[101, 202],
        &[0.1, 0.2],
        &[
            1.0, 2.0, 3.0, // first embedding
            4.0, 5.0, 6.0, // second embedding
        ],
    )
    .unwrap();
    f.close().unwrap();

    let mut reader = TFRecordReader::new(BufReader::new(File::open(&path).unwrap()), true);
    let r1 = reader.read_record().unwrap().expect("first record");
    let r2 = reader.read_record().unwrap().expect("second record");
    assert!(reader.read_record().unwrap().is_none());

    let d1 = EntryDump::decode(r1.as_ref()).unwrap();
    let d2 = EntryDump::decode(r2.as_ref()).unwrap();

    assert_eq!(d1.id, Some(101));
    assert_eq!(d1.num, vec![0.1, 1.0, 2.0, 3.0]);
    assert_eq!(d2.id, Some(202));
    assert_eq!(d2.num, vec![0.2, 4.0, 5.0, 6.0]);
}

#[test]
fn test_writable_file_append_entry_dump_validates_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("invalid_shape.tfrecord");
    let f = WritableFile::new(&path).unwrap();

    let err = f.append_entry_dump(&[1, 2], &[0.1], &[1.0, 2.0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    let err = f
        .append_entry_dump(&[1, 2], &[0.1, 0.2], &[1.0, 2.0, 3.0])
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn test_writable_file_append_after_close_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("closed.txt");
    let f = WritableFile::new(&path).unwrap();
    f.close().unwrap();
    let err = f.append("x").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BrokenPipe);
}
