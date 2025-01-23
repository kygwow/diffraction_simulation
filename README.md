# diffraction_simulation
basic Angular spectrum implementation as a starting point of diffraction simulation

## Field and grid define
e.g. wavelen = 0.5*um

## Source
Plane wave, Spherical wave with incident angle (in degree)

## Optical Element
Spherical lens with fLens, Aperture
Diffuser(Random)

## Heightmap as phase optical element
OPL in phase = heightmap[x,y] * (2*pi/lambda) * (nMaterial-1)

## Propagate along z direction
Angular Spectrum Method

##  off-axis simulation, Alleiviate Aliasing(TBD)
zeropadding, LSASM, Modified Angular Spectrum Method..
