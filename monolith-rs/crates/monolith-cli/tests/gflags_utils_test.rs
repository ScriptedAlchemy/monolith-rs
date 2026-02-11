use monolith_cli::gflags_utils::*;
use std::collections::HashSet;

#[test]
fn test_extract_help_info_basic_and_multiline() {
    let doc = r#"
      :param test_int1: integer 1 for test
      :param test_int2: integer 2 for test
      :param test_str: string for test
                          and test another line
    "#;

    let res = extract_help_info(doc);
    assert_eq!(res.get("test_int1").unwrap(), "integer 1 for test");
    assert_eq!(res.get("test_int2").unwrap(), "integer 2 for test");
    assert_eq!(
        res.get("test_str").unwrap(),
        "string for test and test another line"
    );
}

#[test]
fn test_update_if_default_matches_python_update_semantics() {
    let mut flags = FlagRegistry::default();
    flags.define_int("test_int1", 2);
    flags.define_int("test_int2", 3);

    // Simulate config with defaults (both defaults are 0), but code sets test_int1=1.
    let default_int1 = 0i64;
    let default_int2 = 0i64;
    let mut test_int1 = 1i64; // non-default in code, should not update
    let mut test_int2 = 0i64; // default in code, should update to flag value (3)

    update_if_default(
        &mut test_int1,
        &default_int1,
        &flags.get_int("test_int1").unwrap(),
    );
    update_if_default(
        &mut test_int2,
        &default_int2,
        &flags.get_int("test_int2").unwrap(),
    );

    assert_eq!(test_int1, 1);
    assert_eq!(test_int2, 3);
}

#[test]
fn test_link_dataclass_to_flags_and_update_by_flags_like_behavior() {
    let mut flags = FlagRegistry::default();
    flags.define_string("testflag6", "");

    // This mirrors the Python test where both `v` and `testflag6` link to the same flag.
    let linker = LinkDataclassToFlags::new(&["testflag6"], &[("v", "testflag6")]);
    let struct_fields: HashSet<&'static str> = ["v", "testflag6"].into_iter().collect();
    let meta = linker.build_meta(&struct_fields, &flags).unwrap();

    #[derive(Debug, Clone)]
    struct C {
        v: Option<String>,
        testflag6: Option<String>,
    }

    impl C {
        fn new(flags: &FlagRegistry, meta: &GflagMeta, v: Option<String>) -> Self {
            let mut c = Self { v, testflag6: None };

            // Mirror Python `update_by_flags` patched __init__ behavior:
            // if field is still default (None) and flag is non-default, set it.
            for (field, flag) in &meta.linked_map {
                let f_cur = match field.as_str() {
                    "v" => c.v.as_ref().map(|s| s.as_str()).unwrap_or(""),
                    "testflag6" => c.testflag6.as_ref().map(|s| s.as_str()).unwrap_or(""),
                    _ => "",
                };
                let flag_cur = flags.get_string(flag).unwrap_or("");
                let flag_def = flags.default_string(flag).unwrap_or("");
                if f_cur.is_empty() && flag_cur != flag_def {
                    match field.as_str() {
                        "v" => c.v = Some(flag_cur.to_string()),
                        "testflag6" => c.testflag6 = Some(flag_cur.to_string()),
                        _ => {}
                    }
                }
            }
            c
        }
    }

    // flag default: ""
    flags.set_string("testflag6", "");
    let c = C::new(&flags, &meta, None);
    assert_eq!(c.v, None);
    assert_eq!(c.testflag6, None);

    // flag non-default: "a" should set both fields if default.
    flags.set_string("testflag6", "a");
    let c = C::new(&flags, &meta, None);
    assert_eq!(c.v, Some("a".to_string()));
    assert_eq!(c.testflag6, Some("a".to_string()));

    // if user passes v explicitly, it should not be overwritten, but testflag6 should.
    flags.set_string("testflag6", "b");
    let c = C::new(&flags, &meta, Some("v".to_string()));
    assert_eq!(c.v, Some("v".to_string()));
    assert_eq!(c.testflag6, Some("b".to_string()));
}

#[test]
fn test_link_flag_inheritance_merging() {
    let mut flags = FlagRegistry::default();
    flags.define_string("testflag7", "");
    flags.set_string("testflag7", "a");

    let base_linker = LinkDataclassToFlags::new(&[], &[("v", "testflag7")]);
    let base_fields: HashSet<&'static str> = ["v"].into_iter().collect();
    let base_meta = base_linker.build_meta(&base_fields, &flags).unwrap();

    // "inheritance" by merging base then derived.
    let derived_meta = GflagMeta::merged_from_mro(&[&base_meta]);

    #[derive(Debug, Clone)]
    struct Derived {
        v: Option<String>,
        v2: String,
    }

    impl Derived {
        fn new(flags: &FlagRegistry, meta: &GflagMeta) -> Self {
            let mut d = Self {
                v: None,
                v2: "v2".to_string(),
            };
            for (field, flag) in &meta.linked_map {
                if field == "v" {
                    let flag_cur = flags.get_string(flag).unwrap_or("");
                    let flag_def = flags.default_string(flag).unwrap_or("");
                    if d.v.is_none() && flag_cur != flag_def {
                        d.v = Some(flag_cur.to_string());
                    }
                }
            }
            d
        }
    }

    let d = Derived::new(&flags, &derived_meta);
    assert_eq!(d.v, Some("a".to_string()));
    assert_eq!(d.v2, "v2");
}
