use super::*;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn put16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn literal_object(big: bool, defined: bool) -> Vec<u8> {
    // One external with a string-table name; the section is deliberately empty.
    let (header, width) = if big { (56, 20) } else { (20, 18) };
    let table = header + 40;
    let strings = table + width;
    let symbol = b"__CxxFrameHandler3\0";
    let mut data = vec![0; strings + 4 + symbol.len()];
    if big {
        put16(&mut data, 2, u16::MAX);
        put16(&mut data, 4, 2);
        put16(&mut data, 6, 0xaa64);
        data[12..28].copy_from_slice(&BIGOBJ_CLASS);
        put32(&mut data, 44, 1);
        put32(&mut data, 48, 96);
        put32(&mut data, 52, 1);
    } else {
        put16(&mut data, 0, 0xaa64);
        put16(&mut data, 2, 1);
        put32(&mut data, 8, 60);
        put32(&mut data, 12, 1);
    }
    put32(&mut data, table + 4, 4);
    put16(&mut data, table + 12, u16::from(defined));
    data[table + width - 2] = 2;
    put32(&mut data, strings, 23);
    data[strings + 4..].copy_from_slice(symbol);
    data
}

fn member(archive: &mut Vec<u8>, label: &[u8], data: &[u8]) {
    let mut header = [b' '; 60];
    header[..label.len()].copy_from_slice(label);
    let size = data.len().to_string();
    header[48..48 + size.len()].copy_from_slice(size.as_bytes());
    header[58..].copy_from_slice(b"`\n");
    archive.extend_from_slice(&header);
    archive.extend_from_slice(data);
    if data.len() % 2 != 0 {
        archive.push(b'\n');
    }
}

#[test]
fn native_link_coff_distinguishes_reference_definition_and_raw_byte_decoys() -> TestResult {
    for big in [false, true] {
        for defined in [false, true] {
            let data = literal_object(big, defined);
            assert_eq!(
                inspect(&data, false)?,
                vec![Observation {
                    member_offset: None,
                    symbol: "__CxxFrameHandler3".into(),
                    binding: if defined { "defined" } else { "undefined" },
                }]
            );
        }
    }
    let mut data = literal_object(false, false);
    data[82..100].copy_from_slice(b"ordinary_function\0");
    data.extend_from_slice(b"_CxxThrowException\0");
    assert_eq!(inspect(&data, false)?, Vec::<Observation>::new());
    Ok(())
}

#[test]
fn native_link_coff_archive_checks_every_member_without_claiming_link_selection() -> TestResult {
    let mut data = b"!<arch>\n".to_vec();
    member(&mut data, b"/", b"ignored linker index");
    member(&mut data, b"//", b"arbitrary-long-member.obj/\n");
    let offset = data.len();
    member(&mut data, b"/0", &literal_object(false, true));
    let second = data.len();
    member(&mut data, b"another.obj/", &literal_object(true, false));
    let result = inspect(&data, true)?;
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].member_offset, Some(offset));
    assert_eq!(result[0].binding, "defined");
    assert_eq!(result[1].member_offset, Some(second));
    assert_eq!(result[1].binding, "undefined");
    Ok(())
}

#[test]
fn native_link_coff_short_import_reports_actual_symbol_and_rejects_trailing_bytes() -> TestResult {
    let payload = b"_CxxThrowException\0vcruntime140.dll\0";
    let mut data = vec![0; 20];
    put16(&mut data, 2, u16::MAX);
    put16(&mut data, 6, 0xaa64);
    put32(&mut data, 12, u32::try_from(payload.len())?);
    data.extend_from_slice(payload);
    assert_eq!(
        inspect(&data, false)?,
        vec![Observation {
            member_offset: None,
            symbol: "_CxxThrowException".into(),
            binding: "import",
        }]
    );
    data.push(0);
    assert!(inspect(&data, false).is_err());
    data.pop();
    put16(&mut data, 18, 7 << 2);
    assert!(inspect(&data, false).is_err());

    let alias_payload = b"ordinary_alias\0vcruntime140.dll\0_CxxThrowException\0";
    data.truncate(20);
    put32(&mut data, 12, u32::try_from(alias_payload.len())?);
    put16(&mut data, 18, 4 << 2);
    data.extend_from_slice(alias_payload);
    assert_eq!(
        inspect(&data, false)?,
        vec![Observation {
            member_offset: None,
            symbol: "_CxxThrowException".into(),
            binding: "import-target",
        }]
    );
    Ok(())
}

#[test]
fn native_link_coff_rejects_truncation_foreign_lto_auxiliary_and_invalid_names() {
    let good = literal_object(false, false);
    for length in [0, 2, 19, 59, 77, good.len() - 1] {
        assert!(inspect(&good[..length], false).is_err());
    }
    for (offset, bytes) in [
        (0, vec![0x64, 0x86]),                 // Foreign architecture.
        (8, 20_u32.to_le_bytes().to_vec()),    // Table overlaps headers.
        (12, u32::MAX.to_le_bytes().to_vec()), // Unbounded symbol count.
        (64, 0_u32.to_le_bytes().to_vec()),    // Name points into string header.
        (64, u32::MAX.to_le_bytes().to_vec()), // Name points beyond strings.
        (77, vec![1]),                         // Auxiliary entry beyond count.
        (100, vec![b'x']),                     // Unterminated long symbol.
    ] {
        let mut data = good.clone();
        data[offset..offset + bytes.len()].copy_from_slice(&bytes);
        assert!(inspect(&data, false).is_err(), "offset {offset}");
    }
    assert!(inspect(b"BC\xc0\xdeLLVM bitcode", false).is_err());
    let mut archive = b"!<arch>\n".to_vec();
    member(&mut archive, b"file.obj/", &good);
    archive[8 + 58] = b'x';
    assert!(inspect(&archive, true).is_err());
    assert!(inspect(b"!<thin>\n", true).is_err());
    assert!(inspect(b"!<arch>\n", true).is_err());
}
