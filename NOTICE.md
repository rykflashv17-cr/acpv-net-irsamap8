# Attribution and provenance

Adapted from [HeinzJiao/ACPV-Net](https://github.com/HeinzJiao/ACPV-Net), upstream commit `cf28ca7adbabd653d39a7a3847eb160096fb200b`.
The upstream source license is CC BY-NC-SA 4.0; its license text is retained.
The original Deventer-512 checkpoint is separately marked CC BY-NC 4.0 by its [model card](https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8).
This project adds IRSAMap eight-class preprocessing, loss/classifier adaptation, staged training, evaluation, tracking, and portable entry points. It does not claim authorship of ACPV-Net or IRSAMap.
Third-party source notices remain in their original files. VAE weights are obtained from the official source, not relabeled as our own weights.
IRSAMap original data are published under CC BY-NC-SA 4.0, verified from the ScienceDB structured dataset metadata on 2026-10-15: https://doi.org/10.57760/sciencedb.j00240.00047 . The derived data retain this license, source attribution and a description of all changes. See manifests/source_licenses.json.
