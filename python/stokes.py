#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  stokes.py - Python code for saving trajectories of Stokes parameters to
              files in either CSV or Poincare-compatible [1] format.

References
  [1] "Mapping of Stokes parameter trajectories onto the Poincaré sphere",
      C code repository at Github, https://github.com/hp35/poincare

Created on Sat Dec 28 10:43:46 2024
Copyright (C) 2025 under GPL 3.0, Fredrik Jonsson
"""
import numpy as np
import datetime

def saveStokesParameters(z, s0, s1, s2, s3, filename, fmt="%f", delimiter=" ",
                         header="p", footer="q", saveformat="poincare",
                         tickspacing=0.1, ticklabel=False,
                         ticklabelspacing=0.0, labelzeta=False):
    """
    Save the currently held field envelopes self.uplus and self.uminus as
    corresponding Stokes parameters to a CSV file. This file is typically
    formatted in such a way that it is directly compatible as input to the
    Poincare program for mapping Stokes parameters onto the Poincaré sphere
    for convenient interpretation of the polarization state.

    Parameters
    ----------
    filename : str
        The filename to which the Stokes parameters are to be saved. Should
        include the suffix of choice, typically ".csv" (for CSV) or ".txt"
        or ".dat" (typical for Poincare format).
    fmt : TYPE, optional
        DESCRIPTION. The default is "%f".
    delimiter : TYPE, optional
        DESCRIPTION. The default is " ".
    header : TYPE, optional
        DESCRIPTION. 
        Example: "header":"p b urgt \"LCP\"".
        The default is "p".
    footer : TYPE, optional
        Example: "footer":"q e lrgt \"RCP\"".
        DESCRIPTION. The default is "q".
    saveformat : str, optional
        The format of the saved Stokes parameters. Use saveformat="csv" in
        order to save the entire set as a regular array in a CSV-file, and
        saveformat="poincare" in order to generate a format for direct
        input to the Poincare (https://hp35/poincare/) visualization
        program.
    tickspacing : float, optional
        The spacing between tick marks added to the trajectories,
        applicable only to the saveformat="poincare" option.

    Returns
    -------
    None.
    """
    def tickmark(z, dz, tickspacing):
        """
        Analyze whether a tickmark should be generated at the current position
        or not.

        Parameters
        ----------
        z : float
            Position along trajectory to be mapped.
        dz : float
            Increment of position between all equidistantly spaced
            discrete s-values along trajectory to be mapped.
        tickspacing : float
            The spacing between tickmarks, expressed in units of
            the path length s along the trajectory.

        Returns
        -------
        bool
            If a tickmark is found to be produced, True is returned;
            otherwise False.
        """
        return (abs((z+dz/2.0)%tickspacing) < dz)

    dz = (z[-1]-z[0])/(len(z)-1)  # z-increment between samples along path
    if saveformat == "csv":
        """
        Save Stokes parameters to file in CSV format.
        """
        stokesparams = np.vstack((s1,s2,s3)).transpose()
        kwargs={"fmt":fmt, "delimiter":delimiter, "comments":"",
                "header":header, "footer":footer}
        np.savetxt(filename, stokesparams, **kwargs)
    elif saveformat == "poincare":
        """
        Save Stokes parameters to file in the Poincare format.
        """
        now = datetime.datetime.now()
        with open(filename, "w") as f:
            f.write("p %% %s\n"%(now))
            print("Saving Stokes parameters to %s"%filename)
            for k in range(len(z)):
                f.write("%1.6f %1.6f %1.6f"%(s1[k],s2[k],s3[k]))
                if labelzeta:
                    if tickmark(z[k], dz, tickspacing):
                        f.write(" t l lft \"$z=%1.6f$\"\n"%z[k])
                if tickmark(z[k], dz, tickspacing):
                    if ticklabel: # Apply text label at tickmark
                        if tickmark(z[k], dz, tickspacing):
                            f.write(" t l rgt \"$%1.2f$\"\n"%(z[k]))
                        else:
                            f.write(" t %% \"$s=%1.6f$\"\n"%(z[k]))
                    else:
                        f.write(" t %% \"$s=%1.6f$\"\n"%(z[k]))
                else:
                    f.write(" %% \"$s=%1.6f$\"\n"%(z[k]))
            f.write("q\n")
            f.close()

    return
