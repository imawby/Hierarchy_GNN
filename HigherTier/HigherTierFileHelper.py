import numpy as np
import uproot
import math

from tensorflow.keras.utils import to_categorical

###################################
###################################
parentTrackScore_min = -0.1
parentTrackScore_max = 1.0
parentNuVertexSeparation_min = -100
parentNuVertexSeparation_max = 500
childNuVertexSeparation_min = -100
childNuVertexSeparation_max = 600
parentEndRegionNHits_min = -10
parentEndRegionNHits_max = 80
parentEndRegionNParticles_min = -1
parentEndRegionNParticles_max = 5
parentEndRegionRToWall_min = -10
parentEndRegionRToWall_max = 350
vertexSeparation_min = -50
vertexSeparation_max = 700
separation3D_min = -50
separation3D_max = 700
chargeRatio_min = 0
chargeRatio_max = 100
pidLinkType_min = 0
pidLinkType_max = 25
openingAngle_min = -10
openingAngle_max = 180
trackShowerLinkType_min = -1
trackShowerLinkType_max = 3
###################################
###################################

def readTree(fileNames) :
    
    ###################################
    # To pull out of tree
    ###################################
    # Node variables
    parentTrackScore = []
    parentNuVertexSeparation = []
    childNuVertexSeparation = []
    #parentBraggVariable = []
    parentEndRegionNHits = []
    parentEndRegionNParticles = []
    parentEndRegionRToWall = []
    # Edge information                                                                                                                                                                                                    
    vertexSeparation = []
    #separationU = []
    #separationV = []
    #separationW = []
    separation3D = []
    chargeRatio = []
    pidLinkType = []
    openingAngle = []
    trackShowerLinkType = []
    # Truth
    trueParentChildLink = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            # Node variables
            parentTrackScore.extend(branches['ParentTrackScore'][iEvent])
            parentNuVertexSeparation.extend(branches['ParentNuVertexSeparation'][iEvent])
            childNuVertexSeparation.extend(branches['ChildNuVertexSeparation'][iEvent])
            parentEndRegionNHits.extend(branches['ParentEndRegionNHits'][iEvent])
            parentEndRegionNParticles.extend(branches['ParentEndRegionNParticles'][iEvent])
            parentEndRegionRToWall.extend(branches['ParentEndRegionRToWall'][iEvent])
            # Edge information
            vertexSeparation.extend(branches['VertexSeparation'][iEvent])
            separation3D.extend(branches['Separation3D'][iEvent])
            chargeRatio.extend(branches['ChargeRatio'][iEvent])
            pidLinkType.extend(branches['PIDLinkType'][iEvent])
            openingAngle.extend(branches['OpeningAngle'][iEvent])
            trackShowerLinkType.extend(branches['TrackShowerLinkType'][iEvent]) 
            # Truth 
            trueParentChildLink.extend(branches['TrueParentChildLink'][iEvent])
        
        
    nLinks = len(trueParentChildLink)
        
    print('We have ', str(nLinks), ' links overall!')
        
    ###################################
    # Now turn things into numpy arrays
    ###################################
    # Node variables
    parentTrackScore = np.array(parentTrackScore)
    parentNuVertexSeparation = np.array(parentNuVertexSeparation)
    childNuVertexSeparation = np.array(childNuVertexSeparation)
    parentEndRegionNHits = np.array(parentEndRegionNHits)
    parentEndRegionNParticles = np.array(parentEndRegionNParticles)
    parentEndRegionRToWall = np.array(parentEndRegionRToWall)
    # Edge information                                                                                                                                                                                                    
    vertexSeparation = np.array(vertexSeparation)
    separation3D = np.array(separation3D)
    chargeRatio = np.array(chargeRatio)
    pidLinkType = np.array(pidLinkType)
    openingAngle = np.array(openingAngle)
    trackShowerLinkType = np.array(trackShowerLinkType)
    # Truth 
    trueParentChildLink = np.array(trueParentChildLink, dtype='int64')
            
    ###################################
    # Normalise variables
    ###################################
    normaliseXAxis(parentTrackScore,parentTrackScore_min, parentTrackScore_max)
    normaliseXAxis(parentNuVertexSeparation, parentNuVertexSeparation_min, parentNuVertexSeparation_max)
    normaliseXAxis(childNuVertexSeparation, childNuVertexSeparation_min, childNuVertexSeparation_max)
    normaliseXAxis(parentEndRegionNHits, parentEndRegionNHits_min, parentEndRegionNHits_max)
    normaliseXAxis(parentEndRegionNParticles, parentEndRegionNParticles_min, parentEndRegionNParticles_max)
    normaliseXAxis(parentEndRegionRToWall, parentEndRegionRToWall_min, parentEndRegionRToWall_max)
    normaliseXAxis(vertexSeparation, vertexSeparation_min, vertexSeparation_max)
    normaliseXAxis(separation3D, separation3D_min, separation3D_max)
    normaliseXAxis(chargeRatio, chargeRatio_min, chargeRatio_max)
    normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
    normaliseXAxis(openingAngle, openingAngle_min, openingAngle_max)
    normaliseXAxis(trackShowerLinkType, trackShowerLinkType_min, trackShowerLinkType_max)

    ###################################
    # Reshape
    ###################################
    # Node variables
    parentTrackScore = parentTrackScore.reshape((nLinks, 1))
    parentNuVertexSeparation = parentNuVertexSeparation.reshape((nLinks, 1))
    childNuVertexSeparation = childNuVertexSeparation.reshape((nLinks, 1))
    parentEndRegionNHits = parentEndRegionNHits.reshape((nLinks, 1))
    parentEndRegionNParticles = parentEndRegionNParticles.reshape((nLinks, 1))
    parentEndRegionRToWall = parentEndRegionRToWall.reshape((nLinks, 1))
    # Edge information                                                                                                                                                                                                    
    vertexSeparation = vertexSeparation.reshape((nLinks, 1))
    separation3D = separation3D.reshape((nLinks, 1))
    chargeRatio = chargeRatio.reshape((nLinks, 1))
    pidLinkType = pidLinkType.reshape((nLinks, 1))
    openingAngle = openingAngle.reshape((nLinks, 1))
    trackShowerLinkType = trackShowerLinkType.reshape((nLinks, 1))
    # Truth
    trueParentChildLink = trueParentChildLink.reshape((nLinks, 1))
    
    ###################################
    # Concatenate
    ###################################          
    variables = np.concatenate((parentTrackScore, parentNuVertexSeparation, childNuVertexSeparation, parentEndRegionNHits, parentEndRegionNParticles, parentEndRegionRToWall, vertexSeparation, separation3D, chargeRatio, pidLinkType, openingAngle, trackShowerLinkType), axis=1)

    ###################################
    # Convert truth vector
    ################################### 
    y = trueParentChildLink #to_categorical(trueParentChildLink, 2) # true/false
    
    return nLinks, variables, y

############################################################################################################################################
############################################################################################################################################

def readEvent(arrayDict) :
      
    ###################################
    # Make sure things are numpy arrays
    ###################################
    # Node variables
    parentTrackScore = np.array(arrayDict["parentTrackScore"])
    parentNuVertexSeparation = np.array(arrayDict["parentNuVertexSeparation"])
    childNuVertexSeparation = np.array(arrayDict["childNuVertexSeparation"])
    parentEndRegionNHits = np.array(arrayDict["parentEndRegionNHits"])
    parentEndRegionNParticles = np.array(arrayDict["parentEndRegionNParticles"])
    parentEndRegionRToWall = np.array(arrayDict["parentEndRegionRToWall"])
    # Edge information                                                                                                                                                                                                    
    vertexSeparation = np.array(arrayDict["vertexSeparation"])
    separation3D = np.array(arrayDict["separation3D"])
    chargeRatio = np.array(arrayDict["chargeRatio"])
    pidLinkType = np.array(arrayDict["pidLinkType"])
    openingAngle = np.array(arrayDict["openingAngle"])
    trackShowerLinkType = np.array(arrayDict["trackShowerLinkType"])
    # Truth 
    trueParentChildLink = np.array(arrayDict["trueParentChildLink"], dtype='int64')
    # Reco 
    parentPFPIndices = np.array(arrayDict["parentPFPIndices"], dtype='int64')
    childPFPIndices = np.array(arrayDict["childPFPIndices"], dtype='int64')
    
    nLinks = trueParentChildLink.shape[0]
            
    ###################################
    # Normalise variables
    ###################################
    normaliseXAxis(parentTrackScore,parentTrackScore_min, parentTrackScore_max)
    normaliseXAxis(parentNuVertexSeparation, parentNuVertexSeparation_min, parentNuVertexSeparation_max)
    normaliseXAxis(childNuVertexSeparation, childNuVertexSeparation_min, childNuVertexSeparation_max)
    normaliseXAxis(parentEndRegionNHits, parentEndRegionNHits_min, parentEndRegionNHits_max)
    normaliseXAxis(parentEndRegionNParticles, parentEndRegionNParticles_min, parentEndRegionNParticles_max)
    normaliseXAxis(parentEndRegionRToWall, parentEndRegionRToWall_min, parentEndRegionRToWall_max)
    normaliseXAxis(vertexSeparation, vertexSeparation_min, vertexSeparation_max)
    normaliseXAxis(separation3D, separation3D_min, separation3D_max)
    normaliseXAxis(chargeRatio, chargeRatio_min, chargeRatio_max)
    normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
    normaliseXAxis(openingAngle, openingAngle_min, openingAngle_max)
    normaliseXAxis(trackShowerLinkType, trackShowerLinkType_min, trackShowerLinkType_max)

    ###################################
    # Reshape
    ###################################
    # Node variables
    parentTrackScore = parentTrackScore.reshape((nLinks, 1))
    parentNuVertexSeparation = parentNuVertexSeparation.reshape((nLinks, 1))
    childNuVertexSeparation = childNuVertexSeparation.reshape((nLinks, 1))
    parentEndRegionNHits = parentEndRegionNHits.reshape((nLinks, 1))
    parentEndRegionNParticles = parentEndRegionNParticles.reshape((nLinks, 1))
    parentEndRegionRToWall = parentEndRegionRToWall.reshape((nLinks, 1))
    # Edge information                                                                                                                                                                                                    
    vertexSeparation = vertexSeparation.reshape((nLinks, 1))
    separation3D = separation3D.reshape((nLinks, 1))
    chargeRatio = chargeRatio.reshape((nLinks, 1))
    pidLinkType = pidLinkType.reshape((nLinks, 1))
    openingAngle = openingAngle.reshape((nLinks, 1))
    trackShowerLinkType = trackShowerLinkType.reshape((nLinks, 1))
    # Truth
    trueParentChildLink = trueParentChildLink.reshape((nLinks, 1))
    
    ###################################
    # Concatenate
    ###################################          
    variables = np.concatenate((parentTrackScore, parentNuVertexSeparation, childNuVertexSeparation, parentEndRegionNHits, parentEndRegionNParticles, parentEndRegionRToWall, vertexSeparation, separation3D, chargeRatio, pidLinkType, openingAngle, trackShowerLinkType), axis=1)

    ###################################
    # Convert truth vector
    ################################### 
    y = trueParentChildLink
    
    return variables, y, parentPFPIndices, childPFPIndices

############################################################################################################################################
############################################################################################################################################

def normaliseXAxis(variable, minLimit, maxLimit) :

    interval = math.fabs(minLimit) + math.fabs(maxLimit)
    variable[variable < 0] = minLimit
    variable[variable > maxLimit] = maxLimit
    variable /= interval

############################################################################################################################################
############################################################################################################################################

def getLinkIndex(parentPFPIndices, childPFPIndices, parentPFPIndex, childPFPIndex) :

    for index in range(parentPFPIndices.shape[0]) :
        if ((parentPFPIndices[index] == parentPFPIndex) and (childPFPIndices[index] == childPFPIndex)) :
            return index
            
    return -1
        
        
        
        
        
        
        
        
        
        
        
        

    