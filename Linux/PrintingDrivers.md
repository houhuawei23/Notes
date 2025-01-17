# Printing Drivers

- [how to install hplip on ubuntu linux](https://linuxcapable.com/how-to-install-hplip-on-ubuntu-linux/)
  - `sudo apt install hplip hplip-data hplip-gui hplip-doc`
## Debian12

```bash
# hplip-gui: HP Linux Printing and Imaging - GUI utilities (Qt-based)
sudo apt install hplip-gui

# deps on:
hplip: HP Linux Printing and Imaging
cups: Common Unix Printing System - PPD/driver support, web interface.
```

HPLIP is composed of:

- System services to handle communications with the printers
- HP CUPS backend driver (hp:) with bi-directional communication with
  HP printers (provides printer status feedback to CUPS and enhanced
  HPIJS functionality such as 4-side full-bleed printing support)
- HP CUPS backend driver for sending faxes (hpfax:)
- hpcups CUPS Raster driver to turn rasterized input from the CUPS
  filter chain into the printer's native format (PCL, LIDIL, ...).
  (hpcups is shipped in a separate package)
- HPIJS Ghostscript IJS driver to rasterize output from PostScript(tm)
  files or from any other input format supported by Ghostscript, and
  also for PostScript(tm) to fax conversion support
  (HPIJS is shipped in a separate package)
- Command line utilities to perform printer maintenance, such as
  ink-level monitoring or pen cleaning and calibration
- GUI and command line utility to download data from the photo card
  interfaces in MFP devices
- GUI and command line utilities to interface with the fax functions
- A GUI toolbox to access all these functions in a friendly way
- HPAIO SANE backend (hpaio) for flatbed and Automatic Document Feeder
  (ADF) scanning using MFP devices
