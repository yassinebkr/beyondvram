# Experiment 6 — bare-metal Linux testbench setup

Minimal Debian dual-boot for the OS-tax A/B (`docs/next-experiments.md` #6).
A testbench, not an OS fork: stock netinst, no desktop environment.

## 1. Media

- ISO: `debian-13.6.0-amd64-netinst.iso`, stored on the data HDD (verify
  against `SHA256SUMS` from the Debian mirror before flashing; sha256
  verified 2026-08-14).
- Flash to a USB stick with any raw writer (Rufus in dd mode, balenaEtcher,
  or `dd`). The machine boots UEFI; Secure Boot off for the NVIDIA driver.

## 2. Partition

Route taken on this machine (2026-08-14): the NVMe system partition would not
shrink at all — the shrink engine reported zero shrinkable space
(`Get-PartitionSupportedSize`: SizeMax equal to the current partition size).
The pagefile was relocated to the data HDD and hibernation disabled, but the
Application-log defrag events (source Microsoft-Windows-Defrag, ID 259) then
named the real blocker: `$Mft::$BITMAP` — NTFS metadata the online shrink
engine can never relocate — parked ~52 MiB from the physical end of the
partition (cluster 0xee41033). No Windows-side setting changes this; the only
NVMe route left is an offline resize (GParted Live, or the Debian installer's
own NTFS resize in manual partitioning) after a clean `chkdsk /scan`, with the
usual backup-first caveat. Alternative: install onto a partition carved from
the data HDD instead — cold model loads from the HDD cost ~95–113 s per bench
arm; steady-state tok/s is unaffected (weights are RAM-resident by the time
measurement starts). **Decision (2026-08-14): the HDD route** — the Linux
partition is carved from the data HDD manually; the offline NTFS resize was
judged not worth the filesystem risk for a testbench.

The Debian installer needs unallocated space: delete the freshly created NTFS
volume in diskmgmt.msc (right-click → Delete Volume) before booting the
installer, or point the installer's manual partitioning at it.

For the NVMe route on a machine that does allow shrinking (admin PowerShell):

```powershell
Resize-Partition -DriveLetter <system-drive> -Size <target>
```

If that fails with `StorageWMI 4097` ("shrink size is too big"), query the real
ceiling with `Get-PartitionSupportedSize -DriveLetter <system-drive>` and
shrink to its SizeMax. Persistent caps usually come from the pagefile or
hibernate file near the partition end: `powercfg /h off` plus temporarily
setting no paging file, reboot, retry, then restore.

Debian installs into the freed space: guided partitioning, use largest
continuous free space, all files in one partition, no swap (64 GiB RAM;
a swap partition would only tempt paging during benches).

## 3. Install

- Tasksel: only "standard system utilities" — no desktop.
- Enable `non-free-firmware` + `non-free` repos, then:
  `apt install nvidia-driver firmware-misc-nonfree build-essential cmake git ntfs-3g`
- Windows dual-boot note: no BitLocker on the system partition (checked
  2026-08-14), so GRUB chainloading is uncomplicated; still record the
  recovery-key situation before resizing if that ever changes.

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
