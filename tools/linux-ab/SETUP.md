# Experiment 6 — bare-metal Linux testbench setup

Minimal Debian dual-boot for the OS-tax A/B (`docs/next-experiments.md` #6).
A testbench, not an OS fork: stock netinst, no desktop environment.

## 1. Media

- ISO: `debian-13.6.0-amd64-netinst.iso` (verify against
  `SHA256SUMS` in the same directory before flashing).
- Flash to a USB stick with any raw writer (Rufus in dd mode, balenaEtcher,
  or `dd`). The machine boots UEFI; Secure Boot off for the NVIDIA driver.

## 2. Partition

The system partition on the NVMe is shrunk by ~60 GB (admin PowerShell):

```powershell
Resize-Partition -DriveLetter <system-drive> -Size <target>
```

If the resize fails with `StorageWMI 4097` ("shrink size is too big"), immovable
files near the partition end cap the shrink below 60 GB. Query the real ceiling
with `Get-PartitionSupportedSize -DriveLetter <system-drive>` and shrink to that SizeMax, or
read the cap directly in diskmgmt.msc → Shrink Volume. Persistent caps
usually come from the pagefile or hibernate file near the partition end:
`powercfg /h off` plus temporarily setting no paging file, reboot, retry, then
restore. Fallback: carve the Linux partition from the data HDD instead — first-load
wall time per cold bench arm rises, steady-state tok/s is unaffected (weights
are RAM-resident by the time measurement starts).

Debian installs into the freed space: guided partitioning, use largest
continuous free space, all files in one partition, no swap (64 GiB RAM;
a swap partition would only tempt paging during benches).

## 3. Install

- Tasksel: only "standard system utilities" — no desktop.
- Enable `non-free-firmware` + `non-free` repos, then:
  `apt install nvidia-driver firmware-misc-nonfree build-essential cmake git ntfs-3g`
- Windows dual-boot note: the system partition has no BitLocker (checked 2026-08-14), so GRUB
  chainloading is uncomplicated; still record the recovery-key situation
  before resizing if that ever changes.

## 4. Bench payload

```bash
mkdir -p /opt /root/models
tar xzf llama-b10355-bin-ubuntu-x64.tar.gz -C /opt   # from tools/wsl2-ab/
# Copy models over NTFS (in-kernel ntfs3), then unmount — benches must read
# from ext4 only, never mmap through FUSE/NTFS during a measured run:
mount -t ntfs3 /dev/nvme0n1p3 /mnt/win
cp /mnt/win/Users/yassi/Documents/code/BeyondVram/models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf /root/models/
# optionally the two Qwen3-30B files for the Track-1/#4 arms
umount /mnt/win
./bench_linux.sh            # writes ./linux-ab-results/
```

The WSL2-built Ubuntu binaries run as-is once the NVIDIA driver provides
`libcuda.so`; rebuild from the pinned `dd1ea5243` source only if the
binary refuses to load.

## 5. Protocol

`bench_linux.sh` runs the experiment-2 gpt-oss arms (0,0 / 24,24 / 24,10)
and the Qwen3-30B arms (stock 48,33; mid24-Q3_K variant 48,30), 5 reps each.
Arms cannot interleave across reboots: run ≥3 alternating boot cycles
(Windows → Linux → Windows → …) and compare session means, not single runs.
Native Windows reference numbers come from `results/gpt-oss/placement-grid*.json`,
`thread-sweep.json`, and `results/moe-locality/mixed-precision-speed-q3k.json`.

Decision rule (#6): ≥10% tg advantage on identical configs makes the
dual-boot the bench OS for the 64 GiB / gpt-oss-120b phase.
