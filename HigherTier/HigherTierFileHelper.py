import numpy as np
import uproot
import math

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

###################################
###################################
parentTrackScore_min = -1.0
parentTrackScore_max = 1.0
parentNuVertexSeparation_min = -100
parentNuVertexSeparation_max = 750
childNuVertexSeparation_min = -100
childNuVertexSeparation_max = 750
parentEndRegionNHits_min = -10
parentEndRegionNHits_max = 80
parentEndRegionNParticles_min = -1
parentEndRegionNParticles_max = 5
parentEndRegionRToWall_min = -10
parentEndRegionRToWall_max = 400
vertexSeparation_min = -50
vertexSeparation_max = 700
separation3D_min = -50
separation3D_max = 700
doesChildConnect_min = -1
doesChildConnect_max = 1
overshootDCA_min = -700
overshootDCA_max = 700
overshootL_min = -100
overshootL_max = 700
childConnectionDCA_min = -5
childConnectionDCA_max = 50
childConnectionExtrapDistance_min = -500
childConnectionExtrapDistance_max = 500
childConnectionLRatio_min = -1
childConnectionLRatio_max = 1
parentConnectionPointNUpstreamHits_min = -10 
parentConnectionPointNUpstreamHits_max = 100
parentConnectionPointNDownstreamHits_min = -10
parentConnectionPointNDownstreamHits_max = 100
parentConnectionPointNHitRatio_min = -5
parentConnectionPointNHitRatio_max = 30
parentConnectionPointEigenValueRatio_min = -5 # this seems broken
parentConnectionPointEigenValueRatio_max = 50 # this seems broken
parentConnectionPointOpeningAngle_min = -10
parentConnectionPointOpeningAngle_max = 180
pidLinkType_min = -1
pidLinkType_max = 25
openingAngle_min = -10
openingAngle_max = 180
trackShowerLinkType_min = -2
trackShowerLinkType_max = 3

###################################
###################################

def readTree(fileNames) :
    
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    parentNuVertexSeparation = []
    childNuVertexSeparation = []
    parentEndRegionNHits = []
    parentEndRegionNParticles = []
    parentEndRegionRToWall = []
    vertexSeparation = []
    separation3D = []
    doesChildConnect = []
    overshootStartDCA = []
    overshootStartL = []
    overshootEndDCA = []
    overshootEndL = []
    childConnectionDCA = []
    childConnectionExtrapDistance = []
    childConnectionLRatio = []
    parentConnectionPointNUpstreamHits = []
    parentConnectionPointNDownstreamHits = []
    parentConnectionPointNHitRatio = []
    parentConnectionPointEigenValueRatio = []
    parentConnectionPointOpeningAngle = []
    pidLinkType = []
    pidLinkType_cheat = []
    openingAngle = []
    trackShowerLinkType = []
    # Truth
    isHigherTierTrainingLink = []
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    
    # For my interest
    nLinksMade = []
    signal_nLinksMade = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            # Failed to find a nu vertex?
            if (branches['RecoNuVtxZ'][iEvent] < -900) :
                continue
            
            # Nu vertex vars - to work out separation
            dX = branches['RecoNuVtxX'][iEvent] - branches['NuX'][iEvent]
            dY = branches['RecoNuVtxY'][iEvent] - branches['NuY'][iEvent]
            dZ = branches['RecoNuVtxZ'][iEvent] - branches['NuZ'][iEvent]
            sep = math.sqrt((dX * dX) + (dY * dY) + (dZ * dZ)) 
            
            if (sep > 5.0) :
                continue
                
            print('iEvent:', iEvent)
                
            parentIndex = np.array(branches['ParentPFPIndex'][iEvent])
            childIndex = np.array(branches['ChildPFPIndex'][iEvent])
            linkTruth = np.array(branches['TrueParentChildLink'][iEvent])
            
            if (parentIndex.shape[0] != 0) :
                currentParent = parentIndex[0]
                currentChild = childIndex[0]
                linksMadeCounter = 1

                print('Looping through the links in this event...')
                print('parentIndex.shape[0]', parentIndex.shape[0])

                for iLink in range(1, parentIndex.shape[0]) :

                    if ((currentParent != parentIndex[iLink]) or (currentChild != childIndex[iLink])) :
                        nLinksMade.append(linksMadeCounter)
                        signal_nLinksMade.append(linkTruth[iLink - 1])
                        
                        linksMadeCounter = 0
                        currentParent = parentIndex[iLink]
                        currentChild = childIndex[iLink]

                    linksMadeCounter = linksMadeCounter + 1

                print('Finished...')
            
            # Edge information
            parentTrackScore.extend(branches['ParentTrackScore'][iEvent])
            parentNuVertexSeparation.extend(branches['ParentNuVertexSeparation'][iEvent])
            childNuVertexSeparation.extend(branches['ChildNuVertexSeparation'][iEvent])
            parentEndRegionNHits.extend(branches['ParentEndRegionNHits'][iEvent])
            parentEndRegionNParticles.extend(branches['ParentEndRegionNParticles'][iEvent])
            parentEndRegionRToWall.extend(branches['ParentEndRegionRToWall'][iEvent])
            vertexSeparation.extend(branches['VertexSeparation'][iEvent])
            separation3D.extend(branches['Separation3D'][iEvent])
            doesChildConnect.extend(branches['DoesChildConnect'][iEvent])
            overshootStartDCA.extend(branches['OvershootStartDCA'][iEvent])
            overshootStartL.extend(branches['OvershootStartL'][iEvent])
            overshootEndDCA.extend(branches['OvershootEndDCA'][iEvent])
            overshootEndL.extend(branches['OvershootEndL'][iEvent])
            childConnectionDCA.extend(branches['ChildConnectionDCA'][iEvent])
            childConnectionExtrapDistance.extend(branches['ChildConnectionExtrapDistance'][iEvent])
            childConnectionLRatio.extend(branches['ChildConnectionLRatio'][iEvent])
            parentConnectionPointNUpstreamHits.extend(branches['ParentConnectionPointNUpstreamHits'][iEvent])
            parentConnectionPointNDownstreamHits.extend(branches['ParentConnectionPointNDownstreamHits'][iEvent])
            parentConnectionPointNHitRatio.extend(branches['ParentConnectionPointNHitRatio'][iEvent])
            parentConnectionPointEigenValueRatio.extend(branches['ParentConnectionPointEigenValueRatio'][iEvent])
            parentConnectionPointOpeningAngle.extend(branches['ParentConnectionPointOpeningAngle'][iEvent])
            pidLinkType.extend(branches['PIDLinkType'][iEvent])
            #pidLinkType_cheat.extend(branches['PIDLinkType_cheat'][iEvent])
            #openingAngle.extend(branches['OpeningAngle'][iEvent])
            trackShowerLinkType.extend(branches['TrackShowerLinkType'][iEvent])
            # Truth 
            trueParentChildLink.extend(branches['TrueParentChildLink'][iEvent])
            isHigherTierTrainingLink.extend(branches['IsHigherTierTrainingLink'][iEvent])
            isLinkOrientationCorrect.extend(branches['IsLinkOrientationCorrect'][iEvent])
                    
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
    vertexSeparation = np.array(vertexSeparation)
    separation3D = np.array(separation3D)
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL)
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL)    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance)
    childConnectionLRatio = np.array(childConnectionLRatio)
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits)
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits)
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio)
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio)
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle)
    pidLinkType = np.array(pidLinkType)
    #pidLinkType_cheat = np.array(pidLinkType_cheat)
    #openingAngle = np.array(openingAngle)
    trackShowerLinkType = np.array(trackShowerLinkType)
    # Truth 
    trueParentChildLink = np.array(trueParentChildLink, dtype='int64')
    isHigherTierTrainingLink = np.array(isHigherTierTrainingLink)
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect)
    
    ###################################
    # Only consider training links!
    ###################################
    isHigherTierTrainingMask = (isHigherTierTrainingLink == True)
    parentTrackScore = parentTrackScore[isHigherTierTrainingMask]
    parentNuVertexSeparation = parentNuVertexSeparation[isHigherTierTrainingMask]
    childNuVertexSeparation = childNuVertexSeparation[isHigherTierTrainingMask]
    parentEndRegionNHits = parentEndRegionNHits[isHigherTierTrainingMask]
    parentEndRegionNParticles = parentEndRegionNParticles[isHigherTierTrainingMask]
    parentEndRegionRToWall = parentEndRegionRToWall[isHigherTierTrainingMask]
    vertexSeparation = vertexSeparation[isHigherTierTrainingMask]
    separation3D = separation3D[isHigherTierTrainingMask]
    doesChildConnect = doesChildConnect[isHigherTierTrainingMask]
    overshootStartDCA = overshootStartDCA[isHigherTierTrainingMask]
    overshootStartL = overshootStartL[isHigherTierTrainingMask]
    overshootEndDCA = overshootEndDCA[isHigherTierTrainingMask]
    overshootEndL = overshootEndL[isHigherTierTrainingMask]
    childConnectionDCA = childConnectionDCA[isHigherTierTrainingMask]
    childConnectionExtrapDistance = childConnectionExtrapDistance[isHigherTierTrainingMask]
    childConnectionLRatio = childConnectionLRatio[isHigherTierTrainingMask]
    parentConnectionPointNUpstreamHits = parentConnectionPointNUpstreamHits[isHigherTierTrainingMask]
    parentConnectionPointNDownstreamHits = parentConnectionPointNDownstreamHits[isHigherTierTrainingMask]
    parentConnectionPointNHitRatio = parentConnectionPointNHitRatio[isHigherTierTrainingMask]
    parentConnectionPointEigenValueRatio = parentConnectionPointEigenValueRatio[isHigherTierTrainingMask]
    parentConnectionPointOpeningAngle = parentConnectionPointOpeningAngle[isHigherTierTrainingMask]
    pidLinkType = pidLinkType[isHigherTierTrainingMask]
    #pidLinkType_cheat = pidLinkType_cheat[isHigherTierTrainingMask]
    #openingAngle = openingAngle[isHigherTierTrainingMask]
    trackShowerLinkType = trackShowerLinkType[isHigherTierTrainingMask]
    trueParentChildLink = trueParentChildLink[isHigherTierTrainingMask]
    isHigherTierTrainingLink = isHigherTierTrainingLink[isHigherTierTrainingMask]
    isLinkOrientationCorrect =isLinkOrientationCorrect[isHigherTierTrainingMask]
        
    nLinks = isHigherTierTrainingLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
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
    normaliseXAxis(doesChildConnect, doesChildConnect_min, doesChildConnect_max)
    normaliseXAxis(overshootStartDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootStartL, overshootL_min, overshootL_max)
    normaliseXAxis(overshootEndDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootEndL, overshootL_min, overshootL_max)
    normaliseXAxis(childConnectionDCA, childConnectionDCA_min, childConnectionDCA_max)
    normaliseXAxis(childConnectionExtrapDistance, childConnectionExtrapDistance_min, childConnectionExtrapDistance_max)
    normaliseXAxis(childConnectionLRatio, childConnectionLRatio_min, childConnectionLRatio_max)
    normaliseXAxis(parentConnectionPointNUpstreamHits, parentConnectionPointNUpstreamHits_min, parentConnectionPointNUpstreamHits_max)
    normaliseXAxis(parentConnectionPointNDownstreamHits, parentConnectionPointNDownstreamHits_min, parentConnectionPointNDownstreamHits_max)
    normaliseXAxis(parentConnectionPointNHitRatio, parentConnectionPointNHitRatio_min, parentConnectionPointNHitRatio_max)
    normaliseXAxis(parentConnectionPointEigenValueRatio, parentConnectionPointEigenValueRatio_min, parentConnectionPointEigenValueRatio_max)
    normaliseXAxis(parentConnectionPointOpeningAngle, parentConnectionPointOpeningAngle_min, parentConnectionPointOpeningAngle_max) 
    normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
    ####normaliseXAxis(pidLinkType_cheat, pidLinkType_min, pidLinkType_max)   
    ####normaliseXAxis(openingAngle, openingAngle_min, openingAngle_max)
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
    doesChildConnect = doesChildConnect.reshape((nLinks, 1))
    overshootStartDCA = overshootStartDCA.reshape((nLinks, 1))
    overshootStartL = overshootStartL.reshape((nLinks, 1))
    overshootEndDCA = overshootEndDCA.reshape((nLinks, 1))
    overshootEndL = overshootEndL.reshape((nLinks, 1))
    childConnectionDCA = childConnectionDCA.reshape((nLinks, 1))
    childConnectionExtrapDistance = childConnectionExtrapDistance.reshape((nLinks, 1))
    childConnectionLRatio = childConnectionLRatio.reshape((nLinks, 1))
    parentConnectionPointNUpstreamHits = parentConnectionPointNUpstreamHits.reshape((nLinks, 1))
    parentConnectionPointNDownstreamHits = parentConnectionPointNDownstreamHits.reshape((nLinks, 1))
    parentConnectionPointNHitRatio = parentConnectionPointNHitRatio.reshape((nLinks, 1))
    parentConnectionPointEigenValueRatio = parentConnectionPointEigenValueRatio.reshape((nLinks, 1))
    parentConnectionPointOpeningAngle = parentConnectionPointOpeningAngle.reshape((nLinks, 1))
    pidLinkType = pidLinkType.reshape((nLinks, 1))
    #pidLinkType_cheat = pidLinkType_cheat.reshape((nLinks, 1))
    #openingAngle = openingAngle.reshape((nLinks, 1))
    trackShowerLinkType = trackShowerLinkType.reshape((nLinks, 1))
    # Truth
    trueParentChildLink = trueParentChildLink.reshape((nLinks, 1))
    isLinkOrientationCorrect = isLinkOrientationCorrect.reshape((nLinks, 1))
    
    ###################################
    # Concatenate
    ###################################          
    variables = np.concatenate((parentTrackScore, \
                                parentNuVertexSeparation, \
                                childNuVertexSeparation, \
                                parentEndRegionNHits, \
                                parentEndRegionNParticles, \
                                parentEndRegionRToWall, \
                                vertexSeparation, \
                                separation3D, \
                                doesChildConnect, \
                                overshootStartDCA, \
                                overshootStartL, \
                                overshootEndDCA, \
                                overshootEndL, \
                                childConnectionDCA, \
                                childConnectionExtrapDistance, \
                                childConnectionLRatio, \
                                parentConnectionPointNUpstreamHits, \
                                parentConnectionPointNDownstreamHits, \
                                parentConnectionPointNHitRatio, \
                                parentConnectionPointEigenValueRatio, \
                                parentConnectionPointOpeningAngle, \
                                pidLinkType, \
                                #pidLinkType_cheat, \
                                #openingAngle, \
                                trackShowerLinkType), axis=1)

    ###################################
    # Convert truth vector
    ################################### 
    y = np.zeros(trueParentChildLink.shape)
    y[np.logical_and(trueParentChildLink, isLinkOrientationCorrect)] = 1
    y[np.logical_and(trueParentChildLink, np.logical_not(isLinkOrientationCorrect))] = 2
    #y = to_categorical(y, 3)
    
    return nLinks, variables, y, np.array(nLinksMade), np.array(signal_nLinksMade)

############################################################################################################################################
############################################################################################################################################

# def readEvent(arrayDict) :
      
#     ###################################
#     # Make sure things are numpy arrays
#     ###################################
#     # Node variables
#     parentTrackScore = np.array(arrayDict["parentTrackScore"])
#     parentNuVertexSeparation = np.array(arrayDict["parentNuVertexSeparation"])
#     childNuVertexSeparation = np.array(arrayDict["childNuVertexSeparation"])
#     parentEndRegionNHits = np.array(arrayDict["parentEndRegionNHits"])
#     parentEndRegionNParticles = np.array(arrayDict["parentEndRegionNParticles"])
#     parentEndRegionRToWall = np.array(arrayDict["parentEndRegionRToWall"])
#     # Edge information                                                                                                                                                                                                    
#     vertexSeparation = np.array(arrayDict["vertexSeparation"])
#     separation3D = np.array(arrayDict["separation3D"])
#     chargeRatio = np.array(arrayDict["chargeRatio"])
#     pidLinkType = np.array(arrayDict["pidLinkType"])
#     openingAngle = np.array(arrayDict["openingAngle"])
#     trackShowerLinkType = np.array(arrayDict["trackShowerLinkType"])
#     # Truth 
#     trueParentChildLink = np.array(arrayDict["trueParentChildLink"], dtype='int64')
#     # Reco 
#     parentPFPIndices = np.array(arrayDict["parentPFPIndices"], dtype='int64')
#     childPFPIndices = np.array(arrayDict["childPFPIndices"], dtype='int64')
    
#     nLinks = trueParentChildLink.shape[0]
            
#     ###################################
#     # Normalise variables
#     ###################################
#     normaliseXAxis(parentTrackScore,parentTrackScore_min, parentTrackScore_max)
#     normaliseXAxis(parentNuVertexSeparation, parentNuVertexSeparation_min, parentNuVertexSeparation_max)
#     normaliseXAxis(childNuVertexSeparation, childNuVertexSeparation_min, childNuVertexSeparation_max)
#     normaliseXAxis(parentEndRegionNHits, parentEndRegionNHits_min, parentEndRegionNHits_max)
#     normaliseXAxis(parentEndRegionNParticles, parentEndRegionNParticles_min, parentEndRegionNParticles_max)
#     normaliseXAxis(parentEndRegionRToWall, parentEndRegionRToWall_min, parentEndRegionRToWall_max)
#     normaliseXAxis(vertexSeparation, vertexSeparation_min, vertexSeparation_max)
#     normaliseXAxis(separation3D, separation3D_min, separation3D_max)
#     normaliseXAxis(chargeRatio, chargeRatio_min, chargeRatio_max)
#     normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
#     normaliseXAxis(openingAngle, openingAngle_min, openingAngle_max)
#     normaliseXAxis(trackShowerLinkType, trackShowerLinkType_min, trackShowerLinkType_max)

#     ###################################
#     # Reshape
#     ###################################
#     # Node variables
#     parentTrackScore = parentTrackScore.reshape((nLinks, 1))
#     parentNuVertexSeparation = parentNuVertexSeparation.reshape((nLinks, 1))
#     childNuVertexSeparation = childNuVertexSeparation.reshape((nLinks, 1))
#     parentEndRegionNHits = parentEndRegionNHits.reshape((nLinks, 1))
#     parentEndRegionNParticles = parentEndRegionNParticles.reshape((nLinks, 1))
#     parentEndRegionRToWall = parentEndRegionRToWall.reshape((nLinks, 1))
#     # Edge information                                                                                                                                                                                                    
#     vertexSeparation = vertexSeparation.reshape((nLinks, 1))
#     separation3D = separation3D.reshape((nLinks, 1))
#     chargeRatio = chargeRatio.reshape((nLinks, 1))
#     pidLinkType = pidLinkType.reshape((nLinks, 1))
#     openingAngle = openingAngle.reshape((nLinks, 1))
#     trackShowerLinkType = trackShowerLinkType.reshape((nLinks, 1))
#     # Truth
#     trueParentChildLink = trueParentChildLink.reshape((nLinks, 1))
    
#     ###################################
#     # Concatenate
#     ###################################          
#     variables = np.concatenate((parentTrackScore, parentNuVertexSeparation, childNuVertexSeparation, parentEndRegionNHits, parentEndRegionNParticles, parentEndRegionRToWall, vertexSeparation, separation3D, chargeRatio, pidLinkType, openingAngle, trackShowerLinkType), axis=1)

#     ###################################
#     # Convert truth vector
#     ################################### 
#     y = trueParentChildLink
    
#     return variables, y, parentPFPIndices, childPFPIndices

############################################################################################################################################
############################################################################################################################################

def normaliseXAxis(variable, minLimit, maxLimit) :

    interval = math.fabs(minLimit) + math.fabs(maxLimit)
    variable[variable < minLimit] = minLimit
    variable[variable > maxLimit] = maxLimit
    variable /= interval

############################################################################################################################################
############################################################################################################################################

def getLinkIndex(parentPFPIndices, childPFPIndices, parentPFPIndex, childPFPIndex) :

    for index in range(parentPFPIndices.shape[0]) :
        if ((parentPFPIndices[index] == parentPFPIndex) and (childPFPIndices[index] == childPFPIndex)) :
            return index
            
    return -1
        
############################################################################################################################################
############################################################################################################################################        

def readTreeGroupLinks_track(fileNames) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    parentNuVertexSeparation = []
    childNuVertexSeparation = []
    parentEndRegionNHits = [[], [], [], []]
    parentEndRegionNParticles = [[], [], [], []]
    parentEndRegionRToWall = [[], [], [], []]
    vertexSeparation = [[], [], [], []]
    separation3D = []
    doesChildConnect = [[], [], [], []]
    overshootStartDCA = [[], [], [], []]
    overshootStartL = [[], [], [], []]
    overshootEndDCA = [[], [], [], []]
    overshootEndL = [[], [], [], []]
    childConnectionDCA = [[], [], [], []]
    childConnectionExtrapDistance = [[], [], [], []]
    childConnectionLRatio = [[], [], [], []]
    parentConnectionPointNUpstreamHits = [[], [], [], []]
    parentConnectionPointNDownstreamHits = [[], [], [], []]
    parentConnectionPointNHitRatio = [[], [], [], []]
    parentConnectionPointEigenValueRatio = [[], [], [], []]
    parentConnectionPointOpeningAngle = [[], [], [], []]
    pidLinkType = []
    trackShowerLinkType = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    y = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            # Failed to find a nu vertex?
            if (branches['RecoNuVtxZ'][iEvent] < -900) :
                continue
            
            # Nu vertex vars - to work out separation
            dX = branches['RecoNuVtxX'][iEvent] - branches['NuX'][iEvent]
            dY = branches['RecoNuVtxY'][iEvent] - branches['NuY'][iEvent]
            dZ = branches['RecoNuVtxZ'][iEvent] - branches['NuZ'][iEvent]
            sep = math.sqrt((dX * dX) + (dY * dY) + (dZ * dZ)) 
            
            if (sep > 5.0) :
                continue
                
            ##################################################################
            # DEFINE THEM ALL HERE - apply track-track mask
            ##################################################################
            trackShowerLinkType_file = np.array(branches['TrackShowerLinkType'][iEvent])
            trackShowerLinkType_mask = (trackShowerLinkType_file == 0)

            if (np.count_nonzero(trackShowerLinkType_mask) == 0) :
                continue                
            ##################################################################    
            trackShowerLinkType_file = trackShowerLinkType_file[trackShowerLinkType_mask]
            parentIndex_file = np.array(branches['ParentPFPIndex'][iEvent][trackShowerLinkType_mask])
            childIndex_file = np.array(branches['ChildPFPIndex'][iEvent][trackShowerLinkType_mask])
            parentTrackScore_file = np.array(branches['ParentTrackScore'][iEvent][trackShowerLinkType_mask])
            parentNuVertexSeparation_file = np.array(branches['ParentNuVertexSeparation'][iEvent][trackShowerLinkType_mask])
            childNuVertexSeparation_file = np.array(branches['ChildNuVertexSeparation'][iEvent][trackShowerLinkType_mask])
            parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'][iEvent][trackShowerLinkType_mask])
            parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'][iEvent][trackShowerLinkType_mask])
            parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'][iEvent][trackShowerLinkType_mask])
            vertexSeparation_file = np.array(branches['VertexSeparation'][iEvent][trackShowerLinkType_mask])
            separation3D_file = np.array(branches['Separation3D'][iEvent][trackShowerLinkType_mask])
            doesChildConnect_file = np.array(branches['DoesChildConnect'][iEvent][trackShowerLinkType_mask])
            overshootStartDCA_file = np.array(branches['OvershootStartDCA'][iEvent][trackShowerLinkType_mask])
            overshootStartL_file = np.array(branches['OvershootStartL'][iEvent][trackShowerLinkType_mask])
            overshootEndDCA_file = np.array(branches['OvershootEndDCA'][iEvent][trackShowerLinkType_mask])
            overshootEndL_file = np.array(branches['OvershootEndL'][iEvent][trackShowerLinkType_mask])
            childConnectionDCA_file = np.array(branches['ChildConnectionDCA'][iEvent][trackShowerLinkType_mask])
            childConnectionExtrapDistance_file = np.array(branches['ChildConnectionExtrapDistance'][iEvent][trackShowerLinkType_mask])
            childConnectionLRatio_file = np.array(branches['ChildConnectionLRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNUpstreamHits_file = np.array(branches['ParentConnectionPointNUpstreamHits'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNDownstreamHits_file = np.array(branches['ParentConnectionPointNDownstreamHits'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNHitRatio_file = np.array(branches['ParentConnectionPointNHitRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointEigenValueRatio_file = np.array(branches['ParentConnectionPointEigenValueRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointOpeningAngle_file = np.array(branches['ParentConnectionPointOpeningAngle'][iEvent][trackShowerLinkType_mask])
            pidLinkType_file = np.array(branches['PIDLinkType'][iEvent][trackShowerLinkType_mask])
            trueParentChildLink_file = np.array(branches['TrueParentChildLink'][iEvent][trackShowerLinkType_mask])
            isLinkOrientationCorrect_file = np.array(branches['IsLinkOrientationCorrect'][iEvent][trackShowerLinkType_mask])
            ##################################################################
            # DEFINE THEM ALL HERE - apply generation mask
            ##################################################################
            # I want to discount child primaries from the training
            trainingChild_mask = []
            for childIndex in childIndex_file :
                trainingChild_mask.append(branches['RecoPFPTrueVisibleGeneration'][iEvent][childIndex] != 2)
            
            trainingChild_mask = np.array(trainingChild_mask)
            
            if (np.count_nonzero(trainingChild_mask) == 0) :
                continue
            
            trackShowerLinkType_file = trackShowerLinkType_file[trainingChild_mask]
            parentIndex_file = parentIndex_file[trainingChild_mask]
            childIndex_file = childIndex_file[trainingChild_mask]
            parentTrackScore_file = parentTrackScore_file[trainingChild_mask]
            parentNuVertexSeparation_file = parentNuVertexSeparation_file[trainingChild_mask]
            childNuVertexSeparation_file = childNuVertexSeparation_file[trainingChild_mask]
            parentEndRegionNHits_file = parentEndRegionNHits_file[trainingChild_mask]
            parentEndRegionNParticles_file = parentEndRegionNParticles_file[trainingChild_mask]
            parentEndRegionRToWall_file = parentEndRegionRToWall_file[trainingChild_mask]
            vertexSeparation_file = vertexSeparation_file[trainingChild_mask]
            separation3D_file = separation3D_file[trainingChild_mask]
            doesChildConnect_file = doesChildConnect_file[trainingChild_mask]
            overshootStartDCA_file = overshootStartDCA_file[trainingChild_mask]
            overshootStartL_file = overshootStartL_file[trainingChild_mask]
            overshootEndDCA_file = overshootEndDCA_file[trainingChild_mask]
            overshootEndL_file = overshootEndL_file[trainingChild_mask]
            childConnectionDCA_file = childConnectionDCA_file[trainingChild_mask]
            childConnectionExtrapDistance_file = childConnectionExtrapDistance_file[trainingChild_mask]
            childConnectionLRatio_file = childConnectionLRatio_file[trainingChild_mask]
            parentConnectionPointNUpstreamHits_file = parentConnectionPointNUpstreamHits_file[trainingChild_mask]
            parentConnectionPointNDownstreamHits_file = parentConnectionPointNDownstreamHits_file[trainingChild_mask]
            parentConnectionPointNHitRatio_file = parentConnectionPointNHitRatio_file[trainingChild_mask]
            parentConnectionPointEigenValueRatio_file = parentConnectionPointEigenValueRatio_file[trainingChild_mask]
            parentConnectionPointOpeningAngle_file = parentConnectionPointOpeningAngle_file[trainingChild_mask]
            pidLinkType_file = pidLinkType_file[trainingChild_mask]
            trueParentChildLink_file = trueParentChildLink_file[trainingChild_mask]
            isLinkOrientationCorrect_file = isLinkOrientationCorrect_file[trainingChild_mask]            
            
            ####################################
            # Now loop over loops to group them.
            ####################################
            if (parentIndex_file.shape[0] != 0) :
                currentParent = -1
                currentChild = -1
                linksMadeCounter = 0
                
                this_y = [0, 0, 0, 0]
                this_isLinkOrientationCorrect = [0, 0, 0, 0]
                order = [0, 1, 2, 3]

                for iLink in range(0, parentIndex_file.shape[0]) :

                    # If we have moved onto a new group...
                    if ((currentParent != parentIndex_file[iLink]) or (currentChild != childIndex_file[iLink])) :
                        # set the common vars
                        parentTrackScore.append(parentTrackScore_file[iLink])
                        parentNuVertexSeparation.append(parentNuVertexSeparation_file[iLink])
                        childNuVertexSeparation.append(childNuVertexSeparation_file[iLink])
                        separation3D.append(separation3D_file[iLink])
                        pidLinkType.append(pidLinkType_file[iLink])
                        trackShowerLinkType.append(trackShowerLinkType_file[iLink])
                        trueParentChildLink.append(trueParentChildLink_file[iLink])
                        
                        currentParent = parentIndex_file[iLink]
                        currentChild = childIndex_file[iLink]
                    
                    # Set truth
                    if (trueParentChildLink_file[iLink] and isLinkOrientationCorrect_file[iLink]) :
                        this_y[order[linksMadeCounter]] = 1 
                    elif (trueParentChildLink_file[iLink] and (not isLinkOrientationCorrect_file[iLink])) :
                        this_y[order[linksMadeCounter]] = 2
                        
                    this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_file[iLink]
                    
                    # set the link information
                    parentEndRegionNHits[order[linksMadeCounter]].append(parentEndRegionNHits_file[iLink])
                    parentEndRegionNParticles[order[linksMadeCounter]].append(parentEndRegionNParticles_file[iLink])
                    parentEndRegionRToWall[order[linksMadeCounter]].append(parentEndRegionRToWall_file[iLink])
                    vertexSeparation[order[linksMadeCounter]].append(vertexSeparation_file[iLink])
                    doesChildConnect[order[linksMadeCounter]].append(doesChildConnect_file[iLink])
                    overshootStartDCA[order[linksMadeCounter]].append(overshootStartDCA_file[iLink])
                    overshootStartL[order[linksMadeCounter]].append(overshootStartL_file[iLink])
                    overshootEndDCA[order[linksMadeCounter]].append(overshootEndDCA_file[iLink])
                    overshootEndL[order[linksMadeCounter]].append(overshootEndL_file[iLink])
                    childConnectionDCA[order[linksMadeCounter]].append(childConnectionDCA_file[iLink])
                    childConnectionExtrapDistance[order[linksMadeCounter]].append(childConnectionExtrapDistance_file[iLink])
                    childConnectionLRatio[order[linksMadeCounter]].append(childConnectionLRatio_file[iLink])
                    parentConnectionPointNUpstreamHits[order[linksMadeCounter]].append(parentConnectionPointNUpstreamHits_file[iLink])
                    parentConnectionPointNDownstreamHits[order[linksMadeCounter]].append(parentConnectionPointNDownstreamHits_file[iLink])
                    parentConnectionPointNHitRatio[order[linksMadeCounter]].append(parentConnectionPointNHitRatio_file[iLink])
                    parentConnectionPointEigenValueRatio[order[linksMadeCounter]].append(parentConnectionPointEigenValueRatio_file[iLink])
                    parentConnectionPointOpeningAngle[order[linksMadeCounter]].append(parentConnectionPointOpeningAngle_file[iLink])
                    
                    linksMadeCounter = linksMadeCounter + 1
                    
                    if (linksMadeCounter == 4) :
                        y.append(this_y)
                        isLinkOrientationCorrect.append(this_isLinkOrientationCorrect)
                        
                        # reset                        
                        linksMadeCounter = 0
                        this_y = [0, 0, 0, 0]
                        this_isLinkOrientationCorrect = [0, 0, 0, 0]
                        order = shuffle(order)
             
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
    vertexSeparation = np.array(vertexSeparation)
    separation3D = np.array(separation3D)
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL)
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL)    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance)
    childConnectionLRatio = np.array(childConnectionLRatio)
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits)
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits)
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio)
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio)
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle)
    pidLinkType = np.array(pidLinkType)
    trackShowerLinkType = np.array(trackShowerLinkType)
    # Truth 
    trueParentChildLink = np.array(trueParentChildLink, dtype='int')
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect, dtype='int')
    y = np.array(y, dtype='int')
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = trueParentChildLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
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
    normaliseXAxis(doesChildConnect, doesChildConnect_min, doesChildConnect_max)
    normaliseXAxis(overshootStartDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootStartL, overshootL_min, overshootL_max)
    normaliseXAxis(overshootEndDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootEndL, overshootL_min, overshootL_max)
    normaliseXAxis(childConnectionDCA, childConnectionDCA_min, childConnectionDCA_max)
    normaliseXAxis(childConnectionExtrapDistance, childConnectionExtrapDistance_min, childConnectionExtrapDistance_max)
    normaliseXAxis(childConnectionLRatio, childConnectionLRatio_min, childConnectionLRatio_max)
    normaliseXAxis(parentConnectionPointNUpstreamHits, parentConnectionPointNUpstreamHits_min, parentConnectionPointNUpstreamHits_max)
    normaliseXAxis(parentConnectionPointNDownstreamHits, parentConnectionPointNDownstreamHits_min, parentConnectionPointNDownstreamHits_max)
    normaliseXAxis(parentConnectionPointNHitRatio, parentConnectionPointNHitRatio_min, parentConnectionPointNHitRatio_max)
    normaliseXAxis(parentConnectionPointEigenValueRatio, parentConnectionPointEigenValueRatio_min, parentConnectionPointEigenValueRatio_max)
    normaliseXAxis(parentConnectionPointOpeningAngle, parentConnectionPointOpeningAngle_min, parentConnectionPointOpeningAngle_max) 
    normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
    normaliseXAxis(trackShowerLinkType, trackShowerLinkType_min, trackShowerLinkType_max)

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentEndRegionNHits[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[0, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[0, :].reshape(nLinks, 1), \
                                           vertexSeparation[0, :].reshape(nLinks, 1), \
                                           doesChildConnect[0, :].reshape(nLinks, 1), \
                                           overshootStartDCA[0, :].reshape(nLinks, 1), \
                                           overshootStartL[0, :].reshape(nLinks, 1), \
                                           overshootEndDCA[0, :].reshape(nLinks, 1), \
                                           overshootEndL[0, :].reshape(nLinks, 1), \
                                           childConnectionDCA[0, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentEndRegionNHits[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[1, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[1, :].reshape(nLinks, 1), \
                                           vertexSeparation[1, :].reshape(nLinks, 1), \
                                           doesChildConnect[1, :].reshape(nLinks, 1), \
                                           overshootStartDCA[1, :].reshape(nLinks, 1), \
                                           overshootStartL[1, :].reshape(nLinks, 1), \
                                           overshootEndDCA[1, :].reshape(nLinks, 1), \
                                           overshootEndL[1, :].reshape(nLinks, 1), \
                                           childConnectionDCA[1, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentEndRegionNHits[2, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[2, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[2, :].reshape(nLinks, 1), \
                                           vertexSeparation[2, :].reshape(nLinks, 1), \
                                           doesChildConnect[2, :].reshape(nLinks, 1), \
                                           overshootStartDCA[2, :].reshape(nLinks, 1), \
                                           overshootStartL[2, :].reshape(nLinks, 1), \
                                           overshootEndDCA[2, :].reshape(nLinks, 1), \
                                           overshootEndL[2, :].reshape(nLinks, 1), \
                                           childConnectionDCA[2, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[2, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[2, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[2, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentEndRegionNHits[3, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[3, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[3, :].reshape(nLinks, 1), \
                                           vertexSeparation[3, :].reshape(nLinks, 1), \
                                           doesChildConnect[3, :].reshape(nLinks, 1), \
                                           overshootStartDCA[3, :].reshape(nLinks, 1), \
                                           overshootStartL[3, :].reshape(nLinks, 1), \
                                           overshootEndDCA[3, :].reshape(nLinks, 1), \
                                           overshootEndL[3, :].reshape(nLinks, 1), \
                                           childConnectionDCA[3, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[3, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[3, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[3, :].reshape(nLinks, 1)), axis=1)), axis=1)

    # concatenate variable_single and orientations
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                parentNuVertexSeparation.reshape(nLinks, 1), \
                                childNuVertexSeparation.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                pidLinkType.reshape(nLinks, 1), \
                                trackShowerLinkType.reshape(nLinks, 1), \
                                coc0), axis=1)

    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect

############################################################################################################################################
############################################################################################################################################

def readTreeGroupLinks_shower(fileNames) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    parentNuVertexSeparation = []
    childNuVertexSeparation = []
    parentEndRegionNHits = [[], []]
    parentEndRegionNParticles = [[], []]
    parentEndRegionRToWall = [[], []]
    vertexSeparation = [[], []]
    separation3D = []
    doesChildConnect = [[], []]
    overshootStartDCA = [[], []]
    overshootStartL = [[], []]
    overshootEndDCA = [[], []]
    overshootEndL = [[], []]
    childConnectionDCA = [[], []]
    childConnectionExtrapDistance = [[], []]
    childConnectionLRatio = [[], []]
    parentConnectionPointNUpstreamHits = [[], []]
    parentConnectionPointNDownstreamHits = [[], []]
    parentConnectionPointNHitRatio = [[], []]
    parentConnectionPointEigenValueRatio = [[], []]
    parentConnectionPointOpeningAngle = [[], []]
    pidLinkType = []
    trackShowerLinkType = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    y = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            # Failed to find a nu vertex?
            if (branches['RecoNuVtxZ'][iEvent] < -900) :
                continue
            
            # Nu vertex vars - to work out separation
            dX = branches['RecoNuVtxX'][iEvent] - branches['NuX'][iEvent]
            dY = branches['RecoNuVtxY'][iEvent] - branches['NuY'][iEvent]
            dZ = branches['RecoNuVtxZ'][iEvent] - branches['NuZ'][iEvent]
            sep = math.sqrt((dX * dX) + (dY * dY) + (dZ * dZ)) 
            
            if (sep > 5.0) :
                continue
                
            ##################################################################
            # DEFINE THEM ALL HERE - apply track-track or track-shower mask
            ##################################################################
            trackShowerLinkType_file = np.array(branches['TrackShowerLinkType'][iEvent])
            trackShowerLinkType_mask = (trackShowerLinkType_file == 1)
                
            if (np.count_nonzero(trackShowerLinkType_mask) == 0) :
                continue
            ##################################################################    
            trackShowerLinkType_file = trackShowerLinkType_file[trackShowerLinkType_mask]
            parentIndex_file = np.array(branches['ParentPFPIndex'][iEvent][trackShowerLinkType_mask])
            childIndex_file = np.array(branches['ChildPFPIndex'][iEvent][trackShowerLinkType_mask])
            parentTrackScore_file = np.array(branches['ParentTrackScore'][iEvent][trackShowerLinkType_mask])
            parentNuVertexSeparation_file = np.array(branches['ParentNuVertexSeparation'][iEvent][trackShowerLinkType_mask])
            childNuVertexSeparation_file = np.array(branches['ChildNuVertexSeparation'][iEvent][trackShowerLinkType_mask])
            parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'][iEvent][trackShowerLinkType_mask])
            parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'][iEvent][trackShowerLinkType_mask])
            parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'][iEvent][trackShowerLinkType_mask])
            vertexSeparation_file = np.array(branches['VertexSeparation'][iEvent][trackShowerLinkType_mask])
            separation3D_file = np.array(branches['Separation3D'][iEvent][trackShowerLinkType_mask])
            doesChildConnect_file = np.array(branches['DoesChildConnect'][iEvent][trackShowerLinkType_mask])
            overshootStartDCA_file = np.array(branches['OvershootStartDCA'][iEvent][trackShowerLinkType_mask])
            overshootStartL_file = np.array(branches['OvershootStartL'][iEvent][trackShowerLinkType_mask])
            overshootEndDCA_file = np.array(branches['OvershootEndDCA'][iEvent][trackShowerLinkType_mask])
            overshootEndL_file = np.array(branches['OvershootEndL'][iEvent][trackShowerLinkType_mask])
            childConnectionDCA_file = np.array(branches['ChildConnectionDCA'][iEvent][trackShowerLinkType_mask])
            childConnectionExtrapDistance_file = np.array(branches['ChildConnectionExtrapDistance'][iEvent][trackShowerLinkType_mask])
            childConnectionLRatio_file = np.array(branches['ChildConnectionLRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNUpstreamHits_file = np.array(branches['ParentConnectionPointNUpstreamHits'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNDownstreamHits_file = np.array(branches['ParentConnectionPointNDownstreamHits'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointNHitRatio_file = np.array(branches['ParentConnectionPointNHitRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointEigenValueRatio_file = np.array(branches['ParentConnectionPointEigenValueRatio'][iEvent][trackShowerLinkType_mask])
            parentConnectionPointOpeningAngle_file = np.array(branches['ParentConnectionPointOpeningAngle'][iEvent][trackShowerLinkType_mask])
            pidLinkType_file = np.array(branches['PIDLinkType'][iEvent][trackShowerLinkType_mask])
            trueParentChildLink_file = np.array(branches['TrueParentChildLink'][iEvent][trackShowerLinkType_mask])
            isLinkOrientationCorrect_file = np.array(branches['IsLinkOrientationCorrect'][iEvent][trackShowerLinkType_mask])
            ##################################################################
            # DEFINE THEM ALL HERE - apply generation mask
            ##################################################################
            # I want to discount child primaries from the training
            trainingChild_mask = []
            for childIndex in childIndex_file :
                trainingChild_mask.append(branches['RecoPFPTrueVisibleGeneration'][iEvent][childIndex] != 2)
            
            trainingChild_mask = np.array(trainingChild_mask)
            
            if (np.count_nonzero(trainingChild_mask) == 0) :
                continue
            
            trackShowerLinkType_file = trackShowerLinkType_file[trainingChild_mask]
            parentIndex_file = parentIndex_file[trainingChild_mask]
            childIndex_file = childIndex_file[trainingChild_mask]
            parentTrackScore_file = parentTrackScore_file[trainingChild_mask]
            parentNuVertexSeparation_file = parentNuVertexSeparation_file[trainingChild_mask]
            childNuVertexSeparation_file = childNuVertexSeparation_file[trainingChild_mask]
            parentEndRegionNHits_file = parentEndRegionNHits_file[trainingChild_mask]
            parentEndRegionNParticles_file = parentEndRegionNParticles_file[trainingChild_mask]
            parentEndRegionRToWall_file = parentEndRegionRToWall_file[trainingChild_mask]
            vertexSeparation_file = vertexSeparation_file[trainingChild_mask]
            separation3D_file = separation3D_file[trainingChild_mask]
            doesChildConnect_file = doesChildConnect_file[trainingChild_mask]
            overshootStartDCA_file = overshootStartDCA_file[trainingChild_mask]
            overshootStartL_file = overshootStartL_file[trainingChild_mask]
            overshootEndDCA_file = overshootEndDCA_file[trainingChild_mask]
            overshootEndL_file = overshootEndL_file[trainingChild_mask]
            childConnectionDCA_file = childConnectionDCA_file[trainingChild_mask]
            childConnectionExtrapDistance_file = childConnectionExtrapDistance_file[trainingChild_mask]
            childConnectionLRatio_file = childConnectionLRatio_file[trainingChild_mask]
            parentConnectionPointNUpstreamHits_file = parentConnectionPointNUpstreamHits_file[trainingChild_mask]
            parentConnectionPointNDownstreamHits_file = parentConnectionPointNDownstreamHits_file[trainingChild_mask]
            parentConnectionPointNHitRatio_file = parentConnectionPointNHitRatio_file[trainingChild_mask]
            parentConnectionPointEigenValueRatio_file = parentConnectionPointEigenValueRatio_file[trainingChild_mask]
            parentConnectionPointOpeningAngle_file = parentConnectionPointOpeningAngle_file[trainingChild_mask]
            pidLinkType_file = pidLinkType_file[trainingChild_mask]
            trueParentChildLink_file = trueParentChildLink_file[trainingChild_mask]
            isLinkOrientationCorrect_file = isLinkOrientationCorrect_file[trainingChild_mask]
            
            ####################################
            # Now loop over loops to group them.
            ####################################
            if (parentIndex_file.shape[0] != 0) :
                currentParent = -1
                currentChild = -1
                linksMadeCounter = 0
                
                this_y = [0, 0]
                this_isLinkOrientationCorrect = [0, 0]
                order = [0, 1]

                for iLink in range(0, parentIndex_file.shape[0]) :

                    # If we have moved onto a new group...
                    if ((currentParent != parentIndex_file[iLink]) or (currentChild != childIndex_file[iLink])) :
                        # set the common vars
                        parentTrackScore.append(parentTrackScore_file[iLink])
                        parentNuVertexSeparation.append(parentNuVertexSeparation_file[iLink])
                        childNuVertexSeparation.append(childNuVertexSeparation_file[iLink])
                        separation3D.append(separation3D_file[iLink])
                        pidLinkType.append(pidLinkType_file[iLink])
                        trackShowerLinkType.append(trackShowerLinkType_file[iLink])
                        trueParentChildLink.append(trueParentChildLink_file[iLink])
                        
                        currentParent = parentIndex_file[iLink]
                        currentChild = childIndex_file[iLink]
                    
                    # Set truth
                    if (trueParentChildLink_file[iLink] and isLinkOrientationCorrect_file[iLink]) :
                        this_y[order[linksMadeCounter]] = 1 
                    elif (trueParentChildLink_file[iLink] and (not isLinkOrientationCorrect_file[iLink])) :
                        this_y[order[linksMadeCounter]] = 2

                    this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_file[iLink]
                    
                    # set the link information
                    parentEndRegionNHits[order[linksMadeCounter]].append(parentEndRegionNHits_file[iLink])
                    parentEndRegionNParticles[order[linksMadeCounter]].append(parentEndRegionNParticles_file[iLink])
                    parentEndRegionRToWall[order[linksMadeCounter]].append(parentEndRegionRToWall_file[iLink])
                    vertexSeparation[order[linksMadeCounter]].append(vertexSeparation_file[iLink])
                    doesChildConnect[order[linksMadeCounter]].append(doesChildConnect_file[iLink])
                    overshootStartDCA[order[linksMadeCounter]].append(overshootStartDCA_file[iLink])
                    overshootStartL[order[linksMadeCounter]].append(overshootStartL_file[iLink])
                    overshootEndDCA[order[linksMadeCounter]].append(overshootEndDCA_file[iLink])
                    overshootEndL[order[linksMadeCounter]].append(overshootEndL_file[iLink])
                    childConnectionDCA[order[linksMadeCounter]].append(childConnectionDCA_file[iLink])
                    childConnectionExtrapDistance[order[linksMadeCounter]].append(childConnectionExtrapDistance_file[iLink])
                    childConnectionLRatio[order[linksMadeCounter]].append(childConnectionLRatio_file[iLink])
                    parentConnectionPointNUpstreamHits[order[linksMadeCounter]].append(parentConnectionPointNUpstreamHits_file[iLink])
                    parentConnectionPointNDownstreamHits[order[linksMadeCounter]].append(parentConnectionPointNDownstreamHits_file[iLink])
                    parentConnectionPointNHitRatio[order[linksMadeCounter]].append(parentConnectionPointNHitRatio_file[iLink])
                    parentConnectionPointEigenValueRatio[order[linksMadeCounter]].append(parentConnectionPointEigenValueRatio_file[iLink])
                    parentConnectionPointOpeningAngle[order[linksMadeCounter]].append(parentConnectionPointOpeningAngle_file[iLink])
                    
                    linksMadeCounter = linksMadeCounter + 1
                    
                    if (linksMadeCounter == 2) :
                        y.append(this_y)
                        isLinkOrientationCorrect.append(this_isLinkOrientationCorrect)
                        
                        # reset                        
                        linksMadeCounter = 0
                        this_y = [0, 0]
                        this_isLinkOrientationCorrect = [0, 0]
                        order = shuffle(order)
             
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
    vertexSeparation = np.array(vertexSeparation)
    separation3D = np.array(separation3D)
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL)
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL)    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance)
    childConnectionLRatio = np.array(childConnectionLRatio)
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits)
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits)
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio)
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio)
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle)
    pidLinkType = np.array(pidLinkType)
    trackShowerLinkType = np.array(trackShowerLinkType)
    # Truth 
    trueParentChildLink = np.array(trueParentChildLink, dtype='int')
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect, dtype='int')
    y = np.array(y, dtype='int')
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = trueParentChildLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
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
    normaliseXAxis(doesChildConnect, doesChildConnect_min, doesChildConnect_max)
    normaliseXAxis(overshootStartDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootStartL, overshootL_min, overshootL_max)
    normaliseXAxis(overshootEndDCA, overshootDCA_min, overshootDCA_max)
    normaliseXAxis(overshootEndL, overshootL_min, overshootL_max)
    normaliseXAxis(childConnectionDCA, childConnectionDCA_min, childConnectionDCA_max)
    normaliseXAxis(childConnectionExtrapDistance, childConnectionExtrapDistance_min, childConnectionExtrapDistance_max)
    normaliseXAxis(childConnectionLRatio, childConnectionLRatio_min, childConnectionLRatio_max)
    normaliseXAxis(parentConnectionPointNUpstreamHits, parentConnectionPointNUpstreamHits_min, parentConnectionPointNUpstreamHits_max)
    normaliseXAxis(parentConnectionPointNDownstreamHits, parentConnectionPointNDownstreamHits_min, parentConnectionPointNDownstreamHits_max)
    normaliseXAxis(parentConnectionPointNHitRatio, parentConnectionPointNHitRatio_min, parentConnectionPointNHitRatio_max)
    normaliseXAxis(parentConnectionPointEigenValueRatio, parentConnectionPointEigenValueRatio_min, parentConnectionPointEigenValueRatio_max)
    normaliseXAxis(parentConnectionPointOpeningAngle, parentConnectionPointOpeningAngle_min, parentConnectionPointOpeningAngle_max) 
    normaliseXAxis(pidLinkType, pidLinkType_min, pidLinkType_max)
    normaliseXAxis(trackShowerLinkType, trackShowerLinkType_min, trackShowerLinkType_max)

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentEndRegionNHits[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[0, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[0, :].reshape(nLinks, 1), \
                                           vertexSeparation[0, :].reshape(nLinks, 1), \
                                           doesChildConnect[0, :].reshape(nLinks, 1), \
                                           overshootStartDCA[0, :].reshape(nLinks, 1), \
                                           overshootStartL[0, :].reshape(nLinks, 1), \
                                           overshootEndDCA[0, :].reshape(nLinks, 1), \
                                           overshootEndL[0, :].reshape(nLinks, 1), \
                                           childConnectionDCA[0, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[0, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentEndRegionNHits[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNParticles[1, :].reshape(nLinks, 1), \
                                           parentEndRegionRToWall[1, :].reshape(nLinks, 1), \
                                           vertexSeparation[1, :].reshape(nLinks, 1), \
                                           doesChildConnect[1, :].reshape(nLinks, 1), \
                                           overshootStartDCA[1, :].reshape(nLinks, 1), \
                                           overshootStartL[1, :].reshape(nLinks, 1), \
                                           overshootEndDCA[1, :].reshape(nLinks, 1), \
                                           overshootEndL[1, :].reshape(nLinks, 1), \
                                           childConnectionDCA[1, :].reshape(nLinks, 1), \
                                           childConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           childConnectionLRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNUpstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNDownstreamHits[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointNHitRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointEigenValueRatio[1, :].reshape(nLinks, 1), \
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1)), axis=1)), axis=1)

    # concatenate variable_single and orientations
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                parentNuVertexSeparation.reshape(nLinks, 1), \
                                childNuVertexSeparation.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                pidLinkType.reshape(nLinks, 1), \
                                trackShowerLinkType.reshape(nLinks, 1), \
                                coc0), axis=1)

    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect

############################################################################################################################################
############################################################################################################################################    

def readEvent(eventDict) :
    
    ###################################
    # Need to normalise!
    ###################################    
    normaliseXAxis(eventDict['parentTrackScore'], parentTrackScore_min, parentTrackScore_max)
    normaliseXAxis(eventDict['parentNuVertexSeparation'], parentNuVertexSeparation_min, parentNuVertexSeparation_max)
    normaliseXAxis(eventDict['childNuVertexSeparation'], childNuVertexSeparation_min, childNuVertexSeparation_max)
    normaliseXAxis(eventDict['parentEndRegionNHits'], parentEndRegionNHits_min, parentEndRegionNHits_max)
    normaliseXAxis(eventDict['parentEndRegionNParticles'], parentEndRegionNParticles_min, parentEndRegionNParticles_max)
    normaliseXAxis(eventDict['parentEndRegionRToWall'], parentEndRegionRToWall_min, parentEndRegionRToWall_max)
    normaliseXAxis(eventDict['vertexSeparation'], vertexSeparation_min, vertexSeparation_max)
    normaliseXAxis(eventDict['separation3D'], separation3D_min, separation3D_max)
    normaliseXAxis(eventDict['doesChildConnect'], doesChildConnect_min, doesChildConnect_max)
    normaliseXAxis(eventDict['overshootStartDCA'], overshootDCA_min, overshootDCA_max)
    normaliseXAxis(eventDict['overshootStartL'], overshootL_min, overshootL_max)
    normaliseXAxis(eventDict['overshootEndDCA'], overshootDCA_min, overshootDCA_max)
    normaliseXAxis(eventDict['overshootEndL'], overshootL_min, overshootL_max)
    normaliseXAxis(eventDict['childConnectionDCA'], childConnectionDCA_min, childConnectionDCA_max)
    normaliseXAxis(eventDict['childConnectionExtrapDistance'], childConnectionExtrapDistance_min, childConnectionExtrapDistance_max)
    normaliseXAxis(eventDict['childConnectionLRatio'], childConnectionLRatio_min, childConnectionLRatio_max)
    normaliseXAxis(eventDict['parentConnectionPointNUpstreamHits'], parentConnectionPointNUpstreamHits_min, parentConnectionPointNUpstreamHits_max)
    normaliseXAxis(eventDict['parentConnectionPointNDownstreamHits'], parentConnectionPointNDownstreamHits_min, parentConnectionPointNDownstreamHits_max)
    normaliseXAxis(eventDict['parentConnectionPointNHitRatio'], parentConnectionPointNHitRatio_min, parentConnectionPointNHitRatio_max)
    normaliseXAxis(eventDict['parentConnectionPointEigenValueRatio'], parentConnectionPointEigenValueRatio_min, parentConnectionPointEigenValueRatio_max)
    normaliseXAxis(eventDict['parentConnectionPointOpeningAngle'], parentConnectionPointOpeningAngle_min, parentConnectionPointOpeningAngle_max) 
    normaliseXAxis(eventDict['pidLinkType'], pidLinkType_min, pidLinkType_max)
    normaliseXAxis(eventDict['trackShowerLinkType'], trackShowerLinkType_min, trackShowerLinkType_max)
        
    ###################################
    # track-track links
    ###################################
    # ID
    parentPFPIndex_track = []
    childPFPIndex_track = []
    # Link variables
    parentTrackScore_track = []
    parentNuVertexSeparation_track = []
    childNuVertexSeparation_track = []
    parentEndRegionNHits_track = [[], [], [], []]
    parentEndRegionNParticles_track = [[], [], [], []]
    parentEndRegionRToWall_track = [[], [], [], []]
    vertexSeparation_track = [[], [], [], []]
    separation3D_track = []
    doesChildConnect_track = [[], [], [], []]
    overshootStartDCA_track = [[], [], [], []]
    overshootStartL_track = [[], [], [], []]
    overshootEndDCA_track = [[], [], [], []]
    overshootEndL_track = [[], [], [], []]
    childConnectionDCA_track = [[], [], [], []]
    childConnectionExtrapDistance_track = [[], [], [], []]
    childConnectionLRatio_track = [[], [], [], []]
    parentConnectionPointNUpstreamHits_track = [[], [], [], []]
    parentConnectionPointNDownstreamHits_track = [[], [], [], []]
    parentConnectionPointNHitRatio_track = [[], [], [], []]
    parentConnectionPointEigenValueRatio_track = [[], [], [], []]
    parentConnectionPointOpeningAngle_track = [[], [], [], []]
    pidLinkType_track = []
    trackShowerLinkType_track = []
    # Truth
    trueParentChildLink_track = []
    y_track = []
    ###################################
    # track-shower links
    ###################################
    # ID
    parentPFPIndex_shower = []
    childPFPIndex_shower = []
    # Link variables
    parentTrackScore_shower = []
    parentNuVertexSeparation_shower = []
    childNuVertexSeparation_shower = []
    parentEndRegionNHits_shower = [[], []]
    parentEndRegionNParticles_shower = [[], []]
    parentEndRegionRToWall_shower = [[], []]
    vertexSeparation_shower = [[], []]
    separation3D_shower = []
    doesChildConnect_shower = [[], []]
    overshootStartDCA_shower = [[], []]
    overshootStartL_shower = [[], []]
    overshootEndDCA_shower = [[], []]
    overshootEndL_shower = [[], []]
    childConnectionDCA_shower = [[], []]
    childConnectionExtrapDistance_shower = [[], []]
    childConnectionLRatio_shower = [[], []]
    parentConnectionPointNUpstreamHits_shower = [[], []]
    parentConnectionPointNDownstreamHits_shower = [[], []]
    parentConnectionPointNHitRatio_shower = [[], []]
    parentConnectionPointEigenValueRatio_shower = [[], []]
    parentConnectionPointOpeningAngle_shower = [[], []]
    pidLinkType_shower = []
    trackShowerLinkType_shower = []
    # Truth
    trueParentChildLink_shower = []
    y_shower = []
    
    ####################################
    # Now loop over loops to group them.
    ####################################
    currentParent = -1
    currentChild = -1
    linksMadeCounter = 0                
    this_y_track = [0, 0, 0, 0]
    this_y_shower = [0, 0]

    for iLink in range(eventDict['parentPFPIndex'].shape[0]) :

        trackShowerLinkType_event = eventDict['trackShowerLinkType'][iLink]
        isTrack = (trackShowerLinkType_event > -0.00009) and (trackShowerLinkType_event < 0.1) # because things have been normalised -.-
            
        # If we have moved onto a new group...
        if ((currentParent != eventDict['parentPFPIndex'][iLink]) or (currentChild != eventDict['childPFPIndex'][iLink])) :
            # set the common vars
            parentPFPIndex_track.append(eventDict['parentPFPIndex'][iLink]) if isTrack else \
                parentPFPIndex_shower.append(eventDict['parentPFPIndex'][iLink])
            childPFPIndex_track.append(eventDict['childPFPIndex'][iLink]) if isTrack else \
                childPFPIndex_shower.append(eventDict['childPFPIndex'][iLink])
            
            parentTrackScore_track.append(eventDict['parentTrackScore'][iLink]) if isTrack else \
                parentTrackScore_shower.append(eventDict['parentTrackScore'][iLink])
            parentNuVertexSeparation_track.append(eventDict['parentNuVertexSeparation'][iLink]) if isTrack else \
                parentNuVertexSeparation_shower.append(eventDict['parentNuVertexSeparation'][iLink])
            childNuVertexSeparation_track.append(eventDict['childNuVertexSeparation'][iLink]) if isTrack else \
                childNuVertexSeparation_shower.append(eventDict['childNuVertexSeparation'][iLink])
            separation3D_track.append(eventDict['separation3D'][iLink]) if isTrack else \
                separation3D_shower.append(eventDict['separation3D'][iLink])
            pidLinkType_track.append(eventDict['pidLinkType'][iLink]) if isTrack else \
                pidLinkType_shower.append(eventDict['pidLinkType'][iLink])
            trackShowerLinkType_track.append(eventDict['trackShowerLinkType'][iLink]) if isTrack else \
                trackShowerLinkType_shower.append(eventDict['trackShowerLinkType'][iLink]) 
            trueParentChildLink_track.append(eventDict['trueParentChildLink'][iLink]) if isTrack else \
                trueParentChildLink_shower.append(eventDict['trueParentChildLink'][iLink])
            
            currentParent = eventDict['parentPFPIndex'][iLink]
            currentChild = eventDict['childPFPIndex'][iLink]
                    
        # Set truth
        if (isTrack) :
            if (eventDict['trueParentChildLink'][iLink] and eventDict['isLinkOrientationCorrect'][iLink]) :
                this_y_track[linksMadeCounter] = 1
            elif (eventDict['trueParentChildLink'][iLink] and (not eventDict['isLinkOrientationCorrect'][iLink])) :
                this_y_track[linksMadeCounter] = 2
        else :
            if (eventDict['trueParentChildLink'][iLink] and eventDict['isLinkOrientationCorrect'][iLink]) :
                this_y_shower[linksMadeCounter] = 1
            elif (eventDict['trueParentChildLink'][iLink] and (not eventDict['isLinkOrientationCorrect'][iLink])) :
                this_y_shower[linksMadeCounter] = 2
            
        # set the link information
        parentEndRegionNHits_track[linksMadeCounter].append(eventDict['parentEndRegionNHits'][iLink]) if isTrack else \
            parentEndRegionNHits_shower[linksMadeCounter].append(eventDict['parentEndRegionNHits'][iLink])
        parentEndRegionNParticles_track[linksMadeCounter].append(eventDict['parentEndRegionNParticles'][iLink]) if isTrack else \
            parentEndRegionNParticles_shower[linksMadeCounter].append(eventDict['parentEndRegionNParticles'][iLink])
        parentEndRegionRToWall_track[linksMadeCounter].append(eventDict['parentEndRegionRToWall'][iLink]) if isTrack else \
            parentEndRegionRToWall_shower[linksMadeCounter].append(eventDict['parentEndRegionRToWall'][iLink])
        vertexSeparation_track[linksMadeCounter].append(eventDict['vertexSeparation'][iLink]) if isTrack else \
            vertexSeparation_shower[linksMadeCounter].append(eventDict['vertexSeparation'][iLink])
        doesChildConnect_track[linksMadeCounter].append(eventDict['doesChildConnect'][iLink]) if isTrack else \
            doesChildConnect_shower[linksMadeCounter].append(eventDict['doesChildConnect'][iLink])
        overshootStartDCA_track[linksMadeCounter].append(eventDict['overshootStartDCA'][iLink]) if isTrack else \
            overshootStartDCA_shower[linksMadeCounter].append(eventDict['overshootStartDCA'][iLink])
        overshootStartL_track[linksMadeCounter].append(eventDict['overshootStartL'][iLink]) if isTrack else \
            overshootStartL_shower[linksMadeCounter].append(eventDict['overshootStartL'][iLink])
        overshootEndDCA_track[linksMadeCounter].append(eventDict['overshootEndDCA'][iLink]) if isTrack else \
            overshootEndDCA_shower[linksMadeCounter].append(eventDict['overshootEndDCA'][iLink])
        overshootEndL_track[linksMadeCounter].append(eventDict['overshootEndL'][iLink]) if isTrack else \
            overshootEndL_shower[linksMadeCounter].append(eventDict['overshootEndL'][iLink])
        childConnectionDCA_track[linksMadeCounter].append(eventDict['childConnectionDCA'][iLink]) if isTrack else \
            childConnectionDCA_shower[linksMadeCounter].append(eventDict['childConnectionDCA'][iLink])
        childConnectionExtrapDistance_track[linksMadeCounter].append(eventDict['childConnectionExtrapDistance'][iLink]) if isTrack else \
            childConnectionExtrapDistance_shower[linksMadeCounter].append(eventDict['childConnectionExtrapDistance'][iLink])
        childConnectionLRatio_track[linksMadeCounter].append(eventDict['childConnectionLRatio'][iLink]) if isTrack else \
            childConnectionLRatio_shower[linksMadeCounter].append(eventDict['childConnectionLRatio'][iLink])
        parentConnectionPointNUpstreamHits_track[linksMadeCounter].append(eventDict['parentConnectionPointNUpstreamHits'][iLink]) if isTrack else \
            parentConnectionPointNUpstreamHits_shower[linksMadeCounter].append(eventDict['parentConnectionPointNUpstreamHits'][iLink])
        parentConnectionPointNDownstreamHits_track[linksMadeCounter].append(eventDict['parentConnectionPointNDownstreamHits'][iLink]) if isTrack else \
            parentConnectionPointNDownstreamHits_shower[linksMadeCounter].append(eventDict['parentConnectionPointNDownstreamHits'][iLink])
        parentConnectionPointNHitRatio_track[linksMadeCounter].append(eventDict['parentConnectionPointNHitRatio'][iLink]) if isTrack else \
            parentConnectionPointNHitRatio_shower[linksMadeCounter].append(eventDict['parentConnectionPointNHitRatio'][iLink])
        parentConnectionPointEigenValueRatio_track[linksMadeCounter].append(eventDict['parentConnectionPointEigenValueRatio'][iLink]) if isTrack else \
            parentConnectionPointEigenValueRatio_shower[linksMadeCounter].append(eventDict['parentConnectionPointEigenValueRatio'][iLink])
        parentConnectionPointOpeningAngle_track[linksMadeCounter].append(eventDict['parentConnectionPointOpeningAngle'][iLink]) if isTrack else \
            parentConnectionPointOpeningAngle_shower[linksMadeCounter].append(eventDict['parentConnectionPointOpeningAngle'][iLink])
        
        linksMadeCounter = linksMadeCounter + 1
        
        if (isTrack and (linksMadeCounter == 4)) :
            y_track.append(this_y_track)
            linksMadeCounter = 0
            this_y_track = [0, 0, 0, 0]
            
        if ((not isTrack) and (linksMadeCounter == 2)) :
            y_shower.append(this_y_shower)
            linksMadeCounter = 0
            this_y_shower = [0, 0]
             
    ###################################
    # Now turn things into numpy arrays
    ###################################
    # ID
    parentPFPIndex_track = np.array(parentPFPIndex_track)
    childPFPIndex_track = np.array(childPFPIndex_track)
    # Node variables
    parentTrackScore_track = np.array(parentTrackScore_track)
    parentNuVertexSeparation_track = np.array(parentNuVertexSeparation_track)
    childNuVertexSeparation_track = np.array(childNuVertexSeparation_track)
    parentEndRegionNHits_track = np.array(parentEndRegionNHits_track)
    parentEndRegionNParticles_track = np.array(parentEndRegionNParticles_track)
    parentEndRegionRToWall_track = np.array(parentEndRegionRToWall_track)
    vertexSeparation_track = np.array(vertexSeparation_track)
    separation3D_track = np.array(separation3D_track)
    doesChildConnect_track = np.array(doesChildConnect_track)
    overshootStartDCA_track = np.array(overshootStartDCA_track)
    overshootStartL_track = np.array(overshootStartL_track)
    overshootEndDCA_track = np.array(overshootEndDCA_track)
    overshootEndL_track = np.array(overshootEndL_track)    
    childConnectionDCA_track = np.array(childConnectionDCA_track)
    childConnectionExtrapDistance_track = np.array(childConnectionExtrapDistance_track)
    childConnectionLRatio_track = np.array(childConnectionLRatio_track)
    parentConnectionPointNUpstreamHits_track = np.array(parentConnectionPointNUpstreamHits_track)
    parentConnectionPointNDownstreamHits_track = np.array(parentConnectionPointNDownstreamHits_track)
    parentConnectionPointNHitRatio_track = np.array(parentConnectionPointNHitRatio_track)
    parentConnectionPointEigenValueRatio_track = np.array(parentConnectionPointEigenValueRatio_track)
    parentConnectionPointOpeningAngle_track = np.array(parentConnectionPointOpeningAngle_track)
    pidLinkType_track = np.array(pidLinkType_track)
    trackShowerLinkType_track = np.array(trackShowerLinkType_track)
    # Truth 
    trueParentChildLink_track = np.array(trueParentChildLink_track)
    y_track = np.array(y_track)
    # ID
    parentPFPIndex_shower = np.array(parentPFPIndex_shower)
    childPFPIndex_shower = np.array(childPFPIndex_shower)
    # Node variables
    parentTrackScore_shower = np.array(parentTrackScore_shower)
    parentNuVertexSeparation_shower = np.array(parentNuVertexSeparation_shower)
    childNuVertexSeparation_shower = np.array(childNuVertexSeparation_shower)
    parentEndRegionNHits_shower = np.array(parentEndRegionNHits_shower)
    parentEndRegionNParticles_shower = np.array(parentEndRegionNParticles_shower)
    parentEndRegionRToWall_shower = np.array(parentEndRegionRToWall_shower)
    vertexSeparation_shower = np.array(vertexSeparation_shower)
    separation3D_shower = np.array(separation3D_shower)
    doesChildConnect_shower = np.array(doesChildConnect_shower)
    overshootStartDCA_shower = np.array(overshootStartDCA_shower)
    overshootStartL_shower = np.array(overshootStartL_shower)
    overshootEndDCA_shower = np.array(overshootEndDCA_shower)
    overshootEndL_shower = np.array(overshootEndL_shower)    
    childConnectionDCA_shower = np.array(childConnectionDCA_shower)
    childConnectionExtrapDistance_shower = np.array(childConnectionExtrapDistance_shower)
    childConnectionLRatio_shower = np.array(childConnectionLRatio_shower)
    parentConnectionPointNUpstreamHits_shower = np.array(parentConnectionPointNUpstreamHits_shower)
    parentConnectionPointNDownstreamHits_shower = np.array(parentConnectionPointNDownstreamHits_shower)
    parentConnectionPointNHitRatio_shower = np.array(parentConnectionPointNHitRatio_shower)
    parentConnectionPointEigenValueRatio_shower = np.array(parentConnectionPointEigenValueRatio_shower)
    parentConnectionPointOpeningAngle_shower = np.array(parentConnectionPointOpeningAngle_shower)
    pidLinkType_shower = np.array(pidLinkType_shower)
    trackShowerLinkType_shower = np.array(trackShowerLinkType_shower)
    # Truth 
    trueParentChildLink_shower = np.array(trueParentChildLink_shower)
    y_shower = np.array(y_shower)
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks_track = parentPFPIndex_track.shape[0]   
    nLinks_shower = parentPFPIndex_shower.shape[0]  

    ###################################
    # Concatenate
    ###################################
    coc0_track = np.concatenate((np.concatenate((parentEndRegionNHits_track[0, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNParticles_track[0, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionRToWall_track[0, :].reshape(nLinks_track, 1), \
                                                 vertexSeparation_track[0, :].reshape(nLinks_track, 1), \
                                                 doesChildConnect_track[0, :].reshape(nLinks_track, 1), \
                                                 overshootStartDCA_track[0, :].reshape(nLinks_track, 1), \
                                                 overshootStartL_track[0, :].reshape(nLinks_track, 1), \
                                                 overshootEndDCA_track[0, :].reshape(nLinks_track, 1), \
                                                 overshootEndL_track[0, :].reshape(nLinks_track, 1), \
                                                 childConnectionDCA_track[0, :].reshape(nLinks_track, 1), \
                                                 childConnectionExtrapDistance_track[0, :].reshape(nLinks_track, 1), \
                                                 childConnectionLRatio_track[0, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNUpstreamHits_track[0, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNDownstreamHits_track[0, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNHitRatio_track[0, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointEigenValueRatio_track[0, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointOpeningAngle_track[0, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentEndRegionNHits_track[1, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNParticles_track[1, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionRToWall_track[1, :].reshape(nLinks_track, 1), \
                                                 vertexSeparation_track[1, :].reshape(nLinks_track, 1), \
                                                 doesChildConnect_track[1, :].reshape(nLinks_track, 1), \
                                                 overshootStartDCA_track[1, :].reshape(nLinks_track, 1), \
                                                 overshootStartL_track[1, :].reshape(nLinks_track, 1), \
                                                 overshootEndDCA_track[1, :].reshape(nLinks_track, 1), \
                                                 overshootEndL_track[1, :].reshape(nLinks_track, 1), \
                                                 childConnectionDCA_track[1, :].reshape(nLinks_track, 1), \
                                                 childConnectionExtrapDistance_track[1, :].reshape(nLinks_track, 1), \
                                                 childConnectionLRatio_track[1, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNUpstreamHits_track[1, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNDownstreamHits_track[1, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNHitRatio_track[1, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointEigenValueRatio_track[1, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointOpeningAngle_track[1, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentEndRegionNHits_track[2, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNParticles_track[2, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionRToWall_track[2, :].reshape(nLinks_track, 1), \
                                                 vertexSeparation_track[2, :].reshape(nLinks_track, 1), \
                                                 doesChildConnect_track[2, :].reshape(nLinks_track, 1), \
                                                 overshootStartDCA_track[2, :].reshape(nLinks_track, 1), \
                                                 overshootStartL_track[2, :].reshape(nLinks_track, 1), \
                                                 overshootEndDCA_track[2, :].reshape(nLinks_track, 1), \
                                                 overshootEndL_track[2, :].reshape(nLinks_track, 1), \
                                                 childConnectionDCA_track[2, :].reshape(nLinks_track, 1), \
                                                 childConnectionExtrapDistance_track[2, :].reshape(nLinks_track, 1), \
                                                 childConnectionLRatio_track[2, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNUpstreamHits_track[2, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNDownstreamHits_track[2, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNHitRatio_track[2, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointEigenValueRatio_track[2, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointOpeningAngle_track[2, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentEndRegionNHits_track[3, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNParticles_track[3, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionRToWall_track[3, :].reshape(nLinks_track, 1), \
                                                 vertexSeparation_track[3, :].reshape(nLinks_track, 1), \
                                                 doesChildConnect_track[3, :].reshape(nLinks_track, 1), \
                                                 overshootStartDCA_track[3, :].reshape(nLinks_track, 1), \
                                                 overshootStartL_track[3, :].reshape(nLinks_track, 1), \
                                                 overshootEndDCA_track[3, :].reshape(nLinks_track, 1), \
                                                 overshootEndL_track[3, :].reshape(nLinks_track, 1), \
                                                 childConnectionDCA_track[3, :].reshape(nLinks_track, 1), \
                                                 childConnectionExtrapDistance_track[3, :].reshape(nLinks_track, 1), \
                                                 childConnectionLRatio_track[3, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNUpstreamHits_track[3, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNDownstreamHits_track[3, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointNHitRatio_track[3, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointEigenValueRatio_track[3, :].reshape(nLinks_track, 1), \
                                                 parentConnectionPointOpeningAngle_track[3, :].reshape(nLinks_track, 1)), axis=1)), axis=1)

    coc0_shower = np.concatenate((np.concatenate((parentEndRegionNHits_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionNParticles_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionRToWall_shower[0, :].reshape(nLinks_shower, 1), \
                                                 vertexSeparation_shower[0, :].reshape(nLinks_shower, 1), \
                                                 doesChildConnect_shower[0, :].reshape(nLinks_shower, 1), \
                                                 overshootStartDCA_shower[0, :].reshape(nLinks_shower, 1), \
                                                 overshootStartL_shower[0, :].reshape(nLinks_shower, 1), \
                                                 overshootEndDCA_shower[0, :].reshape(nLinks_shower, 1), \
                                                 overshootEndL_shower[0, :].reshape(nLinks_shower, 1), \
                                                 childConnectionDCA_shower[0, :].reshape(nLinks_shower, 1), \
                                                 childConnectionExtrapDistance_shower[0, :].reshape(nLinks_shower, 1), \
                                                 childConnectionLRatio_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNUpstreamHits_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNDownstreamHits_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNHitRatio_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointEigenValueRatio_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointOpeningAngle_shower[0, :].reshape(nLinks_shower, 1)), axis=1), \
                                 np.concatenate((parentEndRegionNHits_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionNParticles_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionRToWall_shower[1, :].reshape(nLinks_shower, 1), \
                                                 vertexSeparation_shower[1, :].reshape(nLinks_shower, 1), \
                                                 doesChildConnect_shower[1, :].reshape(nLinks_shower, 1), \
                                                 overshootStartDCA_shower[1, :].reshape(nLinks_shower, 1), \
                                                 overshootStartL_shower[1, :].reshape(nLinks_shower, 1), \
                                                 overshootEndDCA_shower[1, :].reshape(nLinks_shower, 1), \
                                                 overshootEndL_shower[1, :].reshape(nLinks_shower, 1), \
                                                 childConnectionDCA_shower[1, :].reshape(nLinks_shower, 1), \
                                                 childConnectionExtrapDistance_shower[1, :].reshape(nLinks_shower, 1), \
                                                 childConnectionLRatio_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNUpstreamHits_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNDownstreamHits_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointNHitRatio_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointEigenValueRatio_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentConnectionPointOpeningAngle_shower[1, :].reshape(nLinks_shower, 1)), axis=1)), axis=1)
    
    # concatenate variable_single and orientations
    variables_track = np.concatenate((parentTrackScore_track.reshape(nLinks_track, 1), \
                                      parentNuVertexSeparation_track.reshape(nLinks_track, 1), \
                                      childNuVertexSeparation_track.reshape(nLinks_track, 1), \
                                      separation3D_track.reshape(nLinks_track, 1), \
                                      pidLinkType_track.reshape(nLinks_track, 1), \
                                      trackShowerLinkType_track.reshape(nLinks_track, 1), \
                                      coc0_track), axis=1)

    variables_shower = np.concatenate((parentTrackScore_shower.reshape(nLinks_shower, 1), \
                                      parentNuVertexSeparation_shower.reshape(nLinks_shower, 1), \
                                      childNuVertexSeparation_shower.reshape(nLinks_shower, 1), \
                                      separation3D_shower.reshape(nLinks_shower, 1), \
                                      pidLinkType_shower.reshape(nLinks_shower, 1), \
                                      trackShowerLinkType_shower.reshape(nLinks_shower, 1), \
                                      coc0_shower), axis=1)

    
    return parentPFPIndex_track, childPFPIndex_track, variables_track, y_track, trueParentChildLink_track, parentPFPIndex_shower, childPFPIndex_shower, variables_shower, y_shower, trueParentChildLink_shower

############################################################################################################################################
############################################################################################################################################
        
        
        
        
        
        

    