//! Parser-only sandbox hook. The native window never invokes this entry point.
#![forbid(unsafe_code)]

pub(crate) fn parse(source: &str) -> Vec<u8> {
    crate::window::markdown_worker_parse(source)
}
