//! Transactional KV publication for one decoder position, including unwinding.
#![forbid(unsafe_code)]

use std::io;

use super::LayerKvCache;
use crate::rust_codec::{KvTarget, rollback_kv_slices};

impl KvTarget for LayerKvCache {
    type Guard = (usize, usize, usize);

    fn commit(&mut self, _: Self::Guard) {}

    fn rollback(&mut self, (keys, values, positions): Self::Guard) {
        rollback_kv_slices(&mut self.k, &mut self.v, (keys, values));
        self.seq_len = positions;
    }
}

pub(super) struct KvAppend<'a>(crate::rust_codec::KvAppend<'a, LayerKvCache>);

impl<'a> KvAppend<'a> {
    pub(super) fn new(
        cache: &'a mut LayerKvCache,
        keys: &[f32],
        values: &[f32],
    ) -> io::Result<Self> {
        let next = cache.seq_len.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Speech KV position overflow.")
        })?;
        cache.k.try_reserve(keys.len()).map_err(io::Error::other)?;
        cache
            .v
            .try_reserve(values.len())
            .map_err(io::Error::other)?;
        let guard = (cache.k.len(), cache.v.len(), cache.seq_len);
        let k_start = cache.k.len();
        let v_start = cache.v.len();
        cache.k.resize(k_start + keys.len(), 0);
        cache.v.resize(v_start + values.len(), 0);
        if !crate::engine::narrow_f32_slice(keys, &mut cache.k[k_start..])
            || !crate::engine::narrow_f32_slice(values, &mut cache.v[v_start..])
        {
            cache.k.truncate(k_start);
            cache.v.truncate(v_start);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Speech KV activation exceeds finite IEEE binary16 range.",
            ));
        }
        cache.seq_len = next;
        Ok(Self(crate::rust_codec::KvAppend::from_guard(cache, guard)))
    }

    pub(super) const fn cache(&self) -> &LayerKvCache {
        self.0.state
    }

    pub(super) fn commit(self) {
        self.0.commit();
    }
}

#[cfg(test)]
mod tests {
    use std::{io, panic::AssertUnwindSafe};

    use super::{KvAppend, LayerKvCache};

    #[test]
    fn decoder_kv_append_rolls_back_on_unwind_and_commits_only_explicitly() -> io::Result<()> {
        let mut cache = LayerKvCache {
            k: vec![0x8000, 0x3400],
            v: vec![0x3800, 0xba00],
            seq_len: 1,
        };
        let interrupted = std::panic::catch_unwind(AssertUnwindSafe(|| -> io::Result<()> {
            let _append = KvAppend::new(&mut cache, &[1.0, 2.0], &[3.0, 4.0])?;
            // Resume an injected fault without invoking the process panic hook.
            std::panic::resume_unwind(Box::new("injected decoder fault"));
        }));
        assert!(interrupted.is_err());
        assert_eq!(cache.k, [0x8000_u16, 0x3400]);
        assert_eq!(cache.v, [0x3800_u16, 0xba00]);
        assert_eq!(cache.seq_len, 1);
        KvAppend::new(&mut cache, &[1.0, 2.0], &[3.0, 4.0])?.commit();
        assert_eq!(cache.k, [0x8000_u16, 0x3400, 0x3c00, 0x4000]);
        assert_eq!(cache.v, [0x3800_u16, 0xba00, 0x4200, 0x4400]);
        assert_eq!(cache.seq_len, 2);
        cache.seq_len = usize::MAX;
        assert!(KvAppend::new(&mut cache, &[5.0], &[6.0]).is_err());
        assert_eq!(cache.k.len(), 4);
        assert_eq!(cache.v.len(), 4);
        assert_eq!(cache.seq_len, usize::MAX);
        Ok(())
    }
}
