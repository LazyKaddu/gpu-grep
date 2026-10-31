| Command | Mean Wall Time [s] | Change | Factor |
|:---|---:|---:|:---|
| `grep -a 'AashishNegi' benchmark/massive_test.log` | 13.814 ± 3.641 |  |  |
| `rg 'AashishNegi' benchmark/massive_test.log` | 13.500 ± 0.536 | -2.3% |  |
| `./target/release/gpu-grep 'AashishNegi' benchmark/massive_test.log` | 34.220 ± 7.752 | +147.7% | (2.5x slower) |
