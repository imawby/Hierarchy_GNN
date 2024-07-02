import math
import numpy as np

#############################
# Normalisation values
#############################

# track shower score
trackShowerLimitLow = -1.0
trackShowerLimitHigh = 1.0
trackShowerInterval = math.fabs(trackShowerLimitLow) + math.fabs(trackShowerLimitHigh)

# nHits
nHitsLimitLow = 0.0
nHitsLimitHigh = 2000.0
nHitsInterval = math.fabs(nHitsLimitLow) + math.fabs(nHitsLimitHigh)

# charge
chargeLimitLow = 0
chargeLimitHigh = 1e6
chargeInterval = math.fabs(chargeLimitLow) + math.fabs(chargeLimitHigh)

# vertex x
vertexXLimitLow = -400
vertexXLimitHigh = 400
vertexXInterval = math.fabs(vertexXLimitLow) + math.fabs(vertexXLimitHigh)

# vertex y
vertexYLimitLow = -1400
vertexYLimitHigh = 1400
vertexYInterval = math.fabs(vertexYLimitLow) + math.fabs(vertexYLimitHigh)

# vertex z
vertexZLimitLow = -100
vertexZLimitHigh = 1500
vertexZInterval = math.fabs(vertexZLimitLow) + math.fabs(vertexZLimitHigh)

# showerDir_vis
showerDirLimitLow = -1.1
showerDirLimitHigh = 1.1
showerDirInterval = math.fabs(showerDirLimitLow) + math.fabs(showerDirLimitHigh)

# ivysaurus
ivysaurusLimitLow = -1.0
ivysaurusLimitHigh = 1.0
ivysaurusInterval = math.fabs(ivysaurusLimitLow) + math.fabs(ivysaurusLimitHigh)

# trackLength
trackLengthLimitLow = -100.0
trackLengthLimitHigh = 1000.0
trackLengthInterval = math.fabs(trackLengthLimitLow) + math.fabs(trackLengthLimitHigh)

# displacement
displacementLimitLow = -10.0
displacementLimitHigh = 100.0
displacementInterval = math.fabs(displacementLimitLow) + math.fabs(displacementLimitHigh)

# dca
dcaLimitLow = -10.0
dcaLimitHigh = 50.0  
dcaInterval = math.fabs(dcaLimitLow) + math.fabs(dcaLimitHigh)

# nuVertexEnergyAsymmetry
nuVertexEnergyAsymmetryLow = -0.5
nuVertexEnergyAsymmetryHigh = 1.0
nuVertexEnergyAsymmetryInterval = math.fabs(nuVertexEnergyAsymmetryLow) + math.fabs(nuVertexEnergyAsymmetryHigh)
    
# nuVertexEnergyWeightedMeanRadialDistance
nuVertexEnergyWeightedMeanRadialDistanceLow = -1.0
nuVertexEnergyWeightedMeanRadialDistanceHigh = 20.0
nuVertexEnergyWeightedMeanRadialDistanceInterval = math.fabs(nuVertexEnergyWeightedMeanRadialDistanceLow) + math.fabs(nuVertexEnergyWeightedMeanRadialDistanceHigh)

############################
# Functions
############################

def normaliseTrackShowerScore(trackShowerScore_np) :
    trackShowerScore_np[trackShowerScore_np < trackShowerLimitLow] = trackShowerLimitLow
    trackShowerScore_np[trackShowerScore_np > trackShowerLimitHigh] = trackShowerLimitHigh
    trackShowerScore_np /= trackShowerInterval
    
def normaliseNHits(nHits_np) :
    nHits_np[nHits_np < nHitsLimitLow] = nHitsLimitLow
    nHits_np[nHits_np > nHitsLimitHigh] = nHitsLimitHigh
    nHits_np /= float(nHitsInterval)
    
def normaliseCharge(charge_np) :
    charge_np[charge_np < chargeLimitLow] = chargeLimitLow
    charge_np[charge_np > chargeLimitHigh] = chargeLimitHigh
    charge_np /= chargeInterval
    
def normalisePositionX(positionX_np) :
    positionX_np[positionX_np < vertexXLimitLow] = vertexXLimitLow
    positionX_np[positionX_np > vertexXLimitHigh] = vertexXLimitHigh
    positionX_np /= vertexXInterval
    
def normalisePositionY(positionY_np) :
    positionY_np[positionY_np < vertexYLimitLow] = vertexYLimitLow
    positionY_np[positionY_np > vertexYLimitHigh] = vertexYLimitHigh
    positionY_np /= vertexYInterval
    
def normalisePositionZ(positionZ_np) :
    positionZ_np[positionZ_np < vertexZLimitLow] = vertexZLimitLow
    positionZ_np[positionZ_np > vertexZLimitHigh] = vertexZLimitHigh
    positionZ_np /= vertexZInterval
    
def normaliseShowerDir(showerDir_np) :
    showerDir_np[showerDir_np < showerDirLimitLow] = showerDirLimitLow
    showerDir_np[showerDir_np > showerDirLimitHigh] = showerDirLimitHigh
    showerDir_np /= showerDirInterval
    
def normaliseIvysaurusScore(ivysaurus_np) :
    ivysaurus_np[ivysaurus_np < ivysaurusLimitLow] = ivysaurusLimitLow
    ivysaurus_np[ivysaurus_np > ivysaurusLimitHigh] = ivysaurusLimitHigh
    ivysaurus_np /= ivysaurusInterval
    
def normaliseTrackLength(trackLength_np) :    
    trackLength_np[trackLength_np < trackLengthLimitLow] = trackLengthLimitLow
    trackLength_np[trackLength_np > trackLengthLimitHigh] = trackLengthLimitHigh
    trackLength_np /= trackLengthInterval
    
def normaliseDisplacement(displacement_np) :
    displacement_np[displacement_np < displacementLimitLow] = displacementLimitLow
    displacement_np[displacement_np > displacementLimitHigh] = displacementLimitHigh
    displacement_np /= displacementInterval
    
def normaliseDCA(dca_np) :
    dca_np[dca_np < dcaLimitLow] = dcaLimitLow
    dca_np[dca_np > dcaLimitHigh] = dcaLimitHigh
    dca_np /= dcaInterval   
    
def normaliseNuVertexEnergyAsymmetry(nuVertexEnergyAsymmetry_np) :
    nuVertexEnergyAsymmetry_np[nuVertexEnergyAsymmetry_np < nuVertexEnergyAsymmetryLow] = nuVertexEnergyAsymmetryLow
    nuVertexEnergyAsymmetry_np[nuVertexEnergyAsymmetry_np > nuVertexEnergyAsymmetryHigh] = nuVertexEnergyAsymmetryHigh
    nuVertexEnergyAsymmetry_np /= nuVertexEnergyAsymmetryInterval
    
def normaliseNuVertexEnergyWeightedMeanRadialDistance(nuVertexEnergyWeightedMeanRadialDistance_np) :
    nuVertexEnergyWeightedMeanRadialDistance_np[nuVertexEnergyWeightedMeanRadialDistance_np < nuVertexEnergyWeightedMeanRadialDistanceLow] = nuVertexEnergyWeightedMeanRadialDistanceLow
    nuVertexEnergyWeightedMeanRadialDistance_np[nuVertexEnergyWeightedMeanRadialDistance_np > nuVertexEnergyWeightedMeanRadialDistanceHigh] = nuVertexEnergyWeightedMeanRadialDistanceHigh
    nuVertexEnergyWeightedMeanRadialDistance_np /= nuVertexEnergyWeightedMeanRadialDistanceInterval    
    
    