import numpy as np
import uproot
import math

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

###################################
###################################
parentTrackScore_min = -1.0
parentTrackScore_max = 1.0
parentNSpacepoints_min = 0.0
parentNSpacepoints_max = 2000.0
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

def readTreeGroupLinks_track(fileNames, normalise) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    childTrackScore = []
    parentNSpacepoints = []
    childNSpacepoints = []
    separation3D = []
    parentNuVertexSep = [[], [], [], []]
    childNuVertexSep = [[], [], [], []]    
    parentEndRegionNHits = [[], [], [], []]
    parentEndRegionNParticles = [[], [], [], []]
    parentEndRegionRToWall = [[], [], [], []]
    vertexSeparation = [[], [], [], []]
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
    parentIsPOIClosestToNu = [[], [], [], []]
    childIsPOIClosestToNu = [[], [], [], []]
    # Training cut variables
    trainingCutSep = []
    trainingCutL = []
    trainingCutT = []
    trainingCutDoesConnect = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    trueChildVisibleGeneration = []
    y = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        ####################################
        # Set tree
        ####################################  
        treeFile = uproot.open(fileName)
        tree = treeFile['LaterTierTrackTrackTree']
        branches = tree.arrays()

        ####################################
        # Set tree branches
        ####################################
        # Network vars
        parentTrackScore_file = np.array(branches['ParentTrackScore'])            
        childTrackScore_file = np.array(branches['ChildTrackScore'])
        parentNSpacepoints_file = np.array(branches['ParentNSpacepoints'])
        childNSpacepoints_file = np.array(branches['ChildNSpacepoints'])
        separation3D_file = np.array(branches['Separation3D'])
        parentNuVertexSep_file = np.array(branches['ParentNuVertexSep'])
        childNuVertexSep_file = np.array(branches['ChildNuVertexSep'])                        
        parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'])
        parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'])
        parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'])
        vertexSeparation_file = np.array(branches['VertexSeparation'])        
        doesChildConnect_file = np.array(branches['DoesChildConnect'])
        overshootStartDCA_file = np.array(branches['OvershootStartDCA'])
        overshootStartL_file = np.array(branches['OvershootStartL'])
        overshootEndDCA_file = np.array(branches['OvershootEndDCA'])
        overshootEndL_file = np.array(branches['OvershootEndL'])
        childConnectionDCA_file = np.array(branches['ChildCPDCA'])
        childConnectionExtrapDistance_file = np.array(branches['ChildCPExtrapDistance'])
        childConnectionLRatio_file = np.array(branches['ChildCPLRatio'])
        parentConnectionPointNUpstreamHits_file = np.array(branches['ParentCPNUpstreamHits'])
        parentConnectionPointNDownstreamHits_file = np.array(branches['ParentCPNDownstreamHits'])
        parentConnectionPointNHitRatio_file = np.array(branches['ParentCPNHitRatio'])
        parentConnectionPointEigenValueRatio_file = np.array(branches['ParentCPEigenvalueRatio'])
        parentConnectionPointOpeningAngle_file = np.array(branches['ParentCPOpeningAngle'])
        isParentPOIClosestToNu_file = np.array(branches['ParentIsPOIClosestToNu'])
        isChildPOIClosestToNu_file = np.array(branches['ChildIsPOIClosestToNu'])
        # Truth
        trueParentChildLink_file = np.array(branches['IsTrueLink'])
        trueChildVisibleGeneration_file = np.array(branches['ChildTrueVisibleGeneration'])
        isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])
        # Training cuts!
        trainingCutL_file = np.array(branches['TrainingCutL'])
        trainingCutT_file = np.array(branches['TrainingCutT'])

        # nLinks
        nLinks_file = trueParentChildLink_file.shape[0]
        
        ####################################
        # Now loop over loops to group them.
        ####################################
        linksMadeCounter = 0
        this_y = [0, 0, 0, 0]
        this_isLinkOrientationCorrect = [0, 0, 0, 0]
        order = [0, 1, 2, 3]

        for iLink in range(0, nLinks_file) :
                                                  
            if ((iLink % 100) == 0) :
                print('iLink:', str(iLink) + '/' + str(nLinks_file)) 
                    
            # Set truth                                                  
            trueParentChildLink_bool = math.isclose(trueParentChildLink_file[iLink], 1.0, rel_tol=0.001)
            isLinkOrientationCorrect_bool = math.isclose(isLinkOrientationCorrect_file[iLink], 1.0, rel_tol=0.001)
                       
            if (trueParentChildLink_bool and isLinkOrientationCorrect_bool) :
                this_y[order[linksMadeCounter]] = 1 
            elif (trueParentChildLink_bool and (not isLinkOrientationCorrect_bool)) :
                this_y[order[linksMadeCounter]] = 2

            this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_bool
             
            # set the link information
            parentNuVertexSep[order[linksMadeCounter]].append(parentNuVertexSep_file[iLink])
            childNuVertexSep[order[linksMadeCounter]].append(childNuVertexSep_file[iLink])
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
            isParentPOIClosestToNu_bool = math.isclose(isParentPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
            parentIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isParentPOIClosestToNu_bool else 0)
            isChildPOIClosestToNu_bool = math.isclose(isChildPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
            childIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isChildPOIClosestToNu_bool else 0)
                    
            # Add in training cuts 
            if (isLinkOrientationCorrect_bool) :                
                trainingCutSep.append(separation3D_file[iLink])
                doesChildConnect_bool = math.isclose(doesChildConnect_file[iLink], 1.0, rel_tol=0.001)
                trainingCutDoesConnect.append(doesChildConnect_bool)
                trainingCutL.append(trainingCutL_file[iLink])
                trainingCutT.append(trainingCutT_file[iLink])                                          

            linksMadeCounter = linksMadeCounter + 1
                    
            if (linksMadeCounter == 4) :
                # set the common vars
                parentTrackScore.append(parentTrackScore_file[iLink])
                childTrackScore.append(childTrackScore_file[iLink])
                parentNSpacepoints.append(parentNSpacepoints_file[iLink])
                childNSpacepoints.append(childNSpacepoints_file[iLink])
                separation3D.append(separation3D_file[iLink])                                                  
                # set truth                                                                          
                trueChildVisibleGeneration.append(trueChildVisibleGeneration_file[iLink])
                y.append(this_y)
                trueParentChildLink.append(trueParentChildLink_bool)
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
    parentTrackScore = np.array(parentTrackScore, dtype='float64')
    childTrackScore = np.array(childTrackScore, dtype='float64')
    parentNSpacepoints = np.array(parentNSpacepoints, dtype='float64')
    childNSpacepoints = np.array(childNSpacepoints, dtype='float64')
    separation3D = np.array(separation3D, dtype='float64')
    parentNuVertexSep = np.array(parentNuVertexSep, dtype='float64')
    childNuVertexSep = np.array(childNuVertexSep, dtype='float64')    
    parentEndRegionNHits = np.array(parentEndRegionNHits, dtype='float64')
    parentEndRegionNParticles = np.array(parentEndRegionNParticles, dtype='float64')
    parentEndRegionRToWall = np.array(parentEndRegionRToWall, dtype='float64')
    vertexSeparation = np.array(vertexSeparation, dtype='float64')    
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL, dtype='float64')
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL, dtype='float64')    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance, dtype='float64')
    childConnectionLRatio = np.array(childConnectionLRatio, dtype='float64')
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits, dtype='float64')
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits, dtype='float64')
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio, dtype='float64')
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio, dtype='float64')
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle, dtype='float64')
    parentIsPOIClosestToNu = np.array(parentIsPOIClosestToNu, dtype='float64')
    childIsPOIClosestToNu = np.array(childIsPOIClosestToNu, dtype='float64')
    # Training cut variables
    trainingCutSep = np.array(trainingCutSep, dtype='float64')
    trainingCutDoesConnect = np.array(trainingCutDoesConnect, dtype='int')
    trainingCutL = np.array(trainingCutL, dtype='float64')
    trainingCutT = np.array(trainingCutT, dtype='float64')
    # Truth 
    trueChildVisibleGeneration = np.array(trueChildVisibleGeneration, dtype='int')
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
    if (normalise) :
        normaliseXAxis(parentTrackScore, parentTrackScore_min, parentTrackScore_max)
        normaliseXAxis(childTrackScore, parentTrackScore_min, parentTrackScore_max)    
        normaliseXAxis(parentNSpacepoints, parentNSpacepoints_min, parentNSpacepoints_max)   
        normaliseXAxis(childNSpacepoints, parentNSpacepoints_min, parentNSpacepoints_max) 
        normaliseXAxis(parentNuVertexSep, parentNuVertexSeparation_min, parentNuVertexSeparation_max) 
        normaliseXAxis(childNuVertexSep, parentNuVertexSeparation_min, parentNuVertexSeparation_max)        
        normaliseXAxis(separation3D, separation3D_min, separation3D_max)                                                  
        normaliseXAxis(parentEndRegionNHits, parentEndRegionNHits_min, parentEndRegionNHits_max)
        normaliseXAxis(parentEndRegionNParticles, parentEndRegionNParticles_min, parentEndRegionNParticles_max)
        normaliseXAxis(parentEndRegionRToWall, parentEndRegionRToWall_min, parentEndRegionRToWall_max)
        normaliseXAxis(vertexSeparation, vertexSeparation_min, vertexSeparation_max)
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

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentNuVertexSep[0, :].reshape(nLinks, 1),
                                           childNuVertexSep[0, :].reshape(nLinks, 1),
                                           parentEndRegionNHits[0, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[1, :].reshape(nLinks, 1), \
                                           childNuVertexSep[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[1, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[1, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[2, :].reshape(nLinks, 1), \
                                           childNuVertexSep[2, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[2, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[2, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[2, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[2, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[3, :].reshape(nLinks, 1), \
                                           childNuVertexSep[3, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[3, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[3, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[3, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[3, :].reshape(nLinks, 1)), axis=1)), axis=1)
    
    # concatenate variable_single and orientations    
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                childTrackScore.reshape(nLinks, 1), \
                                parentNSpacepoints.reshape(nLinks, 1), \
                                childNSpacepoints.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                coc0), axis=1)
    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect, trueChildVisibleGeneration, trainingCutSep, trainingCutDoesConnect, trainingCutL, trainingCutT

############################################################################################################################################
############################################################################################################################################

def readTreeGroupLinks_shower(fileNames, normalise) :

    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    parentTrackScore = []
    childTrackScore = []
    parentNSpacepoints = []
    childNSpacepoints = []
    separation3D = []
    parentNuVertexSep = [[], []]
    childNuVertexSep = [[], []]
    parentEndRegionNHits = [[], []]
    parentEndRegionNParticles = [[], []]
    parentEndRegionRToWall = [[], []]
    vertexSeparation = [[], []]
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
    parentIsPOIClosestToNu = [[], []]
    childIsPOIClosestToNu = [[], []]
    # Training cut variables
    trainingCutSep = []
    trainingCutDoesConnect = []
    trainingCutL = []
    trainingCutT = []
    # Truth
    trueParentChildLink = []
    isLinkOrientationCorrect = []
    y = []
    trueChildVisibleGeneration = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
        
        ####################################
        # Set tree
        ####################################  
        treeFile = uproot.open(fileName)
        tree = treeFile['LaterTierTrackShowerTree']
        branches = tree.arrays()

        ####################################
        # Set tree branches
        ####################################
        # Network vars
        parentTrackScore_file = np.array(branches['ParentTrackScore'])            
        childTrackScore_file = np.array(branches['ChildTrackScore'])
        parentNSpacepoints_file = np.array(branches['ParentNSpacepoints'])
        childNSpacepoints_file = np.array(branches['ChildNSpacepoints'])
        separation3D_file = np.array(branches['Separation3D'])
        parentNuVertexSep_file = np.array(branches['ParentNuVertexSep'])
        childNuVertexSep_file = np.array(branches['ChildNuVertexSep'])                        
        parentEndRegionNHits_file = np.array(branches['ParentEndRegionNHits'])
        parentEndRegionNParticles_file = np.array(branches['ParentEndRegionNParticles'])
        parentEndRegionRToWall_file = np.array(branches['ParentEndRegionRToWall'])
        vertexSeparation_file = np.array(branches['VertexSeparation'])        
        doesChildConnect_file = np.array(branches['DoesChildConnect'])
        overshootStartDCA_file = np.array(branches['OvershootStartDCA'])
        overshootStartL_file = np.array(branches['OvershootStartL'])
        overshootEndDCA_file = np.array(branches['OvershootEndDCA'])
        overshootEndL_file = np.array(branches['OvershootEndL'])
        childConnectionDCA_file = np.array(branches['ChildCPDCA'])
        childConnectionExtrapDistance_file = np.array(branches['ChildCPExtrapDistance'])
        childConnectionLRatio_file = np.array(branches['ChildCPLRatio'])
        parentConnectionPointNUpstreamHits_file = np.array(branches['ParentCPNUpstreamHits'])
        parentConnectionPointNDownstreamHits_file = np.array(branches['ParentCPNDownstreamHits'])
        parentConnectionPointNHitRatio_file = np.array(branches['ParentCPNHitRatio'])
        parentConnectionPointEigenValueRatio_file = np.array(branches['ParentCPEigenvalueRatio'])
        parentConnectionPointOpeningAngle_file = np.array(branches['ParentCPOpeningAngle'])
        isParentPOIClosestToNu_file = np.array(branches['ParentIsPOIClosestToNu'])
        isChildPOIClosestToNu_file = np.array(branches['ChildIsPOIClosestToNu'])
        # Truth
        trueParentChildLink_file = np.array(branches['IsTrueLink'])
        trueVisibleGeneration_file = np.array(branches['ChildTrueVisibleGeneration'])
        isLinkOrientationCorrect_file = np.array(branches['IsOrientationCorrect'])
        # Training cuts!
        trainingCutL_file = np.array(branches['TrainingCutL'])
        trainingCutT_file = np.array(branches['TrainingCutT'])
        # nLinks
        nLinks_file = trueParentChildLink_file.shape[0]
        
        ####################################
        # Now loop over loops to group them.
        ####################################
        linksMadeCounter = 0
        this_y = [0, 0]
        this_isLinkOrientationCorrect = [0, 0]
        order = [0, 1]

        for iLink in range(0, nLinks_file) :
                                                  
            if ((iLink % 100) == 0) :
                print('iLink:', str(iLink) + '/' + str(nLinks_file)) 
                    
            # Set truth                                                  
            trueParentChildLink_bool = math.isclose(trueParentChildLink_file[iLink], 1.0, rel_tol=0.001)
            isLinkOrientationCorrect_bool = math.isclose(isLinkOrientationCorrect_file[iLink], 1.0, rel_tol=0.001)
                       
            if (trueParentChildLink_bool and isLinkOrientationCorrect_bool) :
                this_y[order[linksMadeCounter]] = 1 
            elif (trueParentChildLink_bool and (not isLinkOrientationCorrect_bool)) :
                this_y[order[linksMadeCounter]] = 2

            this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_bool  
             
            # set the link information
            parentNuVertexSep[order[linksMadeCounter]].append(parentNuVertexSep_file[iLink])
            childNuVertexSep[order[linksMadeCounter]].append(childNuVertexSep_file[iLink])
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
            isParentPOIClosestToNu_bool = math.isclose(isParentPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
            parentIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isParentPOIClosestToNu_bool else 0)
            isChildPOIClosestToNu_bool = math.isclose(isChildPOIClosestToNu_file[iLink], 1.0, rel_tol=0.001)
            childIsPOIClosestToNu[order[linksMadeCounter]].append(1 if isChildPOIClosestToNu_bool else 0)
                    
            # Add in training cuts 
            if (isLinkOrientationCorrect_bool) :
                trainingCutSep.append(separation3D_file[iLink])
                doesChildConnect_bool = math.isclose(doesChildConnect_file[iLink], 1.0, rel_tol=0.001)
                trainingCutDoesConnect.append(doesChildConnect_bool)
                trainingCutL.append(trainingCutL_file[iLink])
                trainingCutT.append(trainingCutT_file[iLink])                                              

            linksMadeCounter = linksMadeCounter + 1
                    
            if (linksMadeCounter == 2) :
                # set the common vars
                parentTrackScore.append(parentTrackScore_file[iLink])
                childTrackScore.append(childTrackScore_file[iLink])
                parentNSpacepoints.append(parentNSpacepoints_file[iLink])
                childNSpacepoints.append(childNSpacepoints_file[iLink])
                separation3D.append(separation3D_file[iLink])                                                  
                # set truth                                                                          
                trueChildVisibleGeneration.append(trueVisibleGeneration_file[iLink])        
                y.append(this_y)
                trueParentChildLink.append(trueParentChildLink_bool)
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
    parentTrackScore = np.array(parentTrackScore, dtype='float64')    
    childTrackScore = np.array(childTrackScore, dtype='float64')
    parentNSpacepoints = np.array(parentNSpacepoints, dtype='float64')
    childNSpacepoints = np.array(childNSpacepoints, dtype='float64')
    separation3D = np.array(separation3D, dtype='float64')
    parentNuVertexSep = np.array(parentNuVertexSep, dtype='float64')
    childNuVertexSep = np.array(childNuVertexSep, dtype='float64')    
    parentEndRegionNHits = np.array(parentEndRegionNHits, dtype='float64')
    parentEndRegionNParticles = np.array(parentEndRegionNParticles, dtype='float64')
    parentEndRegionRToWall = np.array(parentEndRegionRToWall, dtype='float64')
    vertexSeparation = np.array(vertexSeparation, dtype='float64')    
    doesChildConnect = np.array(doesChildConnect, dtype='float64')
    overshootStartDCA = np.array(overshootStartDCA, dtype='float64')
    overshootStartL = np.array(overshootStartL, dtype='float64')
    overshootEndDCA = np.array(overshootEndDCA, dtype='float64')
    overshootEndL = np.array(overshootEndL, dtype='float64')    
    childConnectionDCA = np.array(childConnectionDCA, dtype='float64')
    childConnectionExtrapDistance = np.array(childConnectionExtrapDistance, dtype='float64')
    childConnectionLRatio = np.array(childConnectionLRatio, dtype='float64')
    parentConnectionPointNUpstreamHits = np.array(parentConnectionPointNUpstreamHits, dtype='float64')
    parentConnectionPointNDownstreamHits = np.array(parentConnectionPointNDownstreamHits, dtype='float64')
    parentConnectionPointNHitRatio = np.array(parentConnectionPointNHitRatio, dtype='float64')
    parentConnectionPointEigenValueRatio = np.array(parentConnectionPointEigenValueRatio, dtype='float64')
    parentConnectionPointOpeningAngle = np.array(parentConnectionPointOpeningAngle, dtype='float64')
    parentIsPOIClosestToNu = np.array(parentIsPOIClosestToNu, dtype='float64')
    childIsPOIClosestToNu = np.array(childIsPOIClosestToNu, dtype='float64')
    # Training cut variables
    trainingCutSep = np.array(trainingCutSep, dtype='float64')
    trainingCutDoesConnect = np.array(trainingCutDoesConnect, dtype='int')
    trainingCutL = np.array(trainingCutL, dtype='float64')
    trainingCutT = np.array(trainingCutT, dtype='float64')
    # Truth 
    trueChildVisibleGeneration = np.array(trueChildVisibleGeneration, dtype='int')
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
    if (normalise) :
        normaliseXAxis(parentTrackScore, parentTrackScore_min, parentTrackScore_max)
        normaliseXAxis(childTrackScore, parentTrackScore_min, parentTrackScore_max)    
        normaliseXAxis(parentNSpacepoints, parentNSpacepoints_min, parentNSpacepoints_max)   
        normaliseXAxis(childNSpacepoints, parentNSpacepoints_min, parentNSpacepoints_max) 
        normaliseXAxis(parentNuVertexSep, parentNuVertexSeparation_min, parentNuVertexSeparation_max) 
        normaliseXAxis(childNuVertexSep, parentNuVertexSeparation_min, parentNuVertexSeparation_max) 
        normaliseXAxis(separation3D, separation3D_min, separation3D_max)
        normaliseXAxis(parentEndRegionNHits, parentEndRegionNHits_min, parentEndRegionNHits_max)
        normaliseXAxis(parentEndRegionNParticles, parentEndRegionNParticles_min, parentEndRegionNParticles_max)
        normaliseXAxis(parentEndRegionRToWall, parentEndRegionRToWall_min, parentEndRegionRToWall_max)
        normaliseXAxis(vertexSeparation, vertexSeparation_min, vertexSeparation_max)
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

    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((parentNuVertexSep[0, :].reshape(nLinks, 1), \
                                           childNuVertexSep[0, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[0, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[0, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((parentNuVertexSep[1, :].reshape(nLinks, 1), \
                                           childNuVertexSep[1, :].reshape(nLinks, 1), \
                                           parentEndRegionNHits[1, :].reshape(nLinks, 1), \
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
                                           parentConnectionPointOpeningAngle[1, :].reshape(nLinks, 1), \
                                           parentIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           childIsPOIClosestToNu[1, :].reshape(nLinks, 1)), axis=1)), axis=1)

    # concatenate variable_single and orientations
    variables = np.concatenate((parentTrackScore.reshape(nLinks, 1), \
                                childTrackScore.reshape(nLinks, 1), \
                                parentNSpacepoints.reshape(nLinks, 1), \
                                childNSpacepoints.reshape(nLinks, 1), \
                                separation3D.reshape(nLinks, 1), \
                                coc0), axis=1)
    
    return nLinks, variables, y, trueParentChildLink, isLinkOrientationCorrect, trueChildVisibleGeneration, trainingCutSep, trainingCutDoesConnect, trainingCutL, trainingCutT

############################################################################################################################################
############################################################################################################################################    

def readEvent(eventDict) :
    
    separation3D_notNorm = eventDict['separation3D'].copy()
    
    ###################################
    # Need to normalise!
    ###################################    
    normaliseXAxis(eventDict['parentTrackScore'], parentTrackScore_min, parentTrackScore_max)
    normaliseXAxis(eventDict['childTrackScore'], parentTrackScore_min, parentTrackScore_max)    
    normaliseXAxis(eventDict['parentNSpacepoints'], parentNSpacepoints_min, parentNSpacepoints_max)   
    normaliseXAxis(eventDict['childNSpacepoints'], parentNSpacepoints_min, parentNSpacepoints_max) 
    normaliseXAxis(eventDict['parentNuVertexSep'], parentNuVertexSeparation_min, parentNuVertexSeparation_max) 
    normaliseXAxis(eventDict['childNuVertexSep'], parentNuVertexSeparation_min, parentNuVertexSeparation_max)        
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
    #normaliseXAxis(eventDict['trackShowerLinkType'], trackShowerLinkType_min, trackShowerLinkType_max)
        
    # To establish orientation of particles wrt the neutrino vertex
    recoNuVertex = np.array([eventDict['recoNuX'], eventDict['recoNuY'], eventDict['recoNuZ']])
        
    ###################################
    # track-track links
    ###################################
    # ID
    parentPFPIndex_track = []
    childPFPIndex_track = []
    # Link variables
    parentTrackScore_track = []    
    childTrackScore_track = []
    parentNSpacepoints_track = []
    childNSpacepoints_track = []
    parentNuVertexSep_track = [[], [], [], []]
    childNuVertexSep_track = [[], [], [], []]    
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
    parentIsPOIClosestToNu_track = [[], [], [], []]
    childIsPOIClosestToNu_track = [[], [], [], []]
    # Training cut variables
    trainingCutSep_track = []
    trainingCutL_track = []
    trainingCutT_track = []
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
    childTrackScore_shower = []
    parentNSpacepoints_shower = []
    childNSpacepoints_shower = []
    parentNuVertexSep_shower = [[], []]
    childNuVertexSep_shower = [[], []]         
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
    parentIsPOIClosestToNu_shower = [[], []]
    childIsPOIClosestToNu_shower = [[], []]
    # Training cut variables
    trainingCutSep_shower = []
    trainingCutL_shower = []
    trainingCutT_shower = []
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
        isTrack = (trackShowerLinkType_event == 0)
            
        # If we have moved onto a new group...
        if ((currentParent != eventDict['parentPFPIndex'][iLink]) or (currentChild != eventDict['childPFPIndex'][iLink])) :
            
            # Make sure that it has all of its orientations in the tree
            if isTrack and len(np.where(np.logical_and(eventDict['parentPFPIndex'] == eventDict['parentPFPIndex'][iLink], eventDict['childPFPIndex'] == eventDict['childPFPIndex'][iLink]))[0]) != 4 :
                continue
                
            # Make sure that it has all of its orientations in the tree
            if (not isTrack) and len(np.where(np.logical_and(eventDict['parentPFPIndex'] == eventDict['parentPFPIndex'][iLink], eventDict['childPFPIndex'] == eventDict['childPFPIndex'][iLink]))[0]) != 2 :
                continue
            
            # set the common vars
            parentPFPIndex_track.append(eventDict['parentPFPIndex'][iLink]) if isTrack else \
                parentPFPIndex_shower.append(eventDict['parentPFPIndex'][iLink])
            childPFPIndex_track.append(eventDict['childPFPIndex'][iLink]) if isTrack else \
                childPFPIndex_shower.append(eventDict['childPFPIndex'][iLink])            
            parentTrackScore_track.append(eventDict['parentTrackScore'][iLink]) if isTrack else \
                parentTrackScore_shower.append(eventDict['parentTrackScore'][iLink])            
            childTrackScore_track.append(eventDict['childTrackScore'][eventDict['childPFPIndex'][iLink]]) if isTrack else \
                childTrackScore_shower.append(eventDict['childTrackScore'][eventDict['childPFPIndex'][iLink]])           
            parentNSpacepoints_track.append(eventDict['parentNSpacepoints'][iLink]) if isTrack else \
                parentNSpacepoints_shower.append(eventDict['parentNSpacepoints'][iLink])            
            childNSpacepoints_track.append(eventDict['childNSpacepoints'][iLink]) if isTrack else \
                childNSpacepoints_shower.append(eventDict['childNSpacepoints'][iLink])                                                                          
            separation3D_track.append(eventDict['separation3D'][iLink]) if isTrack else \
                separation3D_shower.append(eventDict['separation3D'][iLink])
            pidLinkType_track.append(eventDict['pidLinkType'][iLink]) if isTrack else \
                pidLinkType_shower.append(eventDict['pidLinkType'][iLink])
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
        parentNuVertexSep_track[linksMadeCounter].append(eventDict['parentNuVertexSep'][iLink]) if isTrack else \
            parentNuVertexSep_shower[linksMadeCounter].append(eventDict['parentNuVertexSep'][iLink])            
        childNuVertexSep_track[linksMadeCounter].append(eventDict['childNuVertexSep'][iLink]) if isTrack else \
            childNuVertexSep_shower[linksMadeCounter].append(eventDict['childNuVertexSep'][iLink])            
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
        
        # Calculate orientation wrt the neutrino vertex
        parentIsPOIClosestToNu_track[linksMadeCounter].append(1 if eventDict['isParentPOIClosestToNu'][iLink] else 0) if isTrack else \
            parentIsPOIClosestToNu_shower[linksMadeCounter].append(1 if eventDict['isParentPOIClosestToNu'][iLink] else 0)
        childIsPOIClosestToNu_track[linksMadeCounter].append(1 if eventDict['isChildPOIClosestToNu'][iLink] else 0) if isTrack else \
            childIsPOIClosestToNu_shower[linksMadeCounter].append(1 if eventDict['isChildPOIClosestToNu'][iLink] else 0)
        
        # Calculate training cuts 
        if (eventDict['isLinkOrientationCorrect'][iLink]) :            
            trainingCutSep_track.append(separation3D_notNorm[iLink]) if isTrack else trainingCutSep_shower.append(separation3D_notNorm[iLink])
            trainingCutL_track.append(eventDict['trainingCutL'][iLink]) if isTrack else trainingCutL_shower.append(eventDict['trainingCutL'][iLink])
            trainingCutT_track.append(eventDict['trainingCutT'][iLink]) if isTrack else trainingCutT_shower.append(eventDict['trainingCutT'][iLink])
        
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
    childTrackScore_track = np.array(childTrackScore_track)
    parentNSpacepoints_track = np.array(parentNSpacepoints_track)
    childNSpacepoints_track = np.array(childNSpacepoints_track)
    parentNuVertexSep_track = np.array(parentNuVertexSep_track)
    childNuVertexSep_track = np.array(childNuVertexSep_track)        
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
    parentIsPOIClosestToNu_track = np.array(parentIsPOIClosestToNu_track)
    childIsPOIClosestToNu_track = np.array(childIsPOIClosestToNu_track)
    pidLinkType_track = np.array(pidLinkType_track)
    # Training cut variables
    trainingCutSep_track = np.array(trainingCutSep_track)
    trainingCutL_track = np.array(trainingCutL_track)
    trainingCutT_track = np.array(trainingCutT_track)
    # Truth 
    trueParentChildLink_track = np.array(trueParentChildLink_track)
    y_track = np.array(y_track)
    # ID
    parentPFPIndex_shower = np.array(parentPFPIndex_shower)
    childPFPIndex_shower = np.array(childPFPIndex_shower)
    # Node variables
    parentTrackScore_shower = np.array(parentTrackScore_shower)    
    childTrackScore_shower = np.array(childTrackScore_shower)
    parentNSpacepoints_shower = np.array(parentNSpacepoints_shower)
    childNSpacepoints_shower = np.array(childNSpacepoints_shower)
    parentNuVertexSep_shower = np.array(parentNuVertexSep_shower)
    childNuVertexSep_shower = np.array(childNuVertexSep_shower)        
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
    parentIsPOIClosestToNu_shower = np.array(parentIsPOIClosestToNu_shower)
    childIsPOIClosestToNu_shower = np.array(childIsPOIClosestToNu_shower)
    pidLinkType_shower = np.array(pidLinkType_shower)
    # Training cut variables
    trainingCutSep_shower = np.array(trainingCutSep_shower)
    trainingCutL_shower = np.array(trainingCutL_shower)
    trainingCutT_shower = np.array(trainingCutT_shower)
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
    coc0_track = np.concatenate((np.concatenate((parentNuVertexSep_track[0, :].reshape(nLinks_track, 1), \
                                                 childNuVertexSep_track[0, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNHits_track[0, :].reshape(nLinks_track, 1), \
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
                                                 parentConnectionPointOpeningAngle_track[0, :].reshape(nLinks_track, 1), \
                                                 parentIsPOIClosestToNu_track[0, :].reshape(nLinks_track, 1), \
                                                 childIsPOIClosestToNu_track[0, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentNuVertexSep_track[1, :].reshape(nLinks_track, 1), \
                                                 childNuVertexSep_track[1, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNHits_track[1, :].reshape(nLinks_track, 1), \
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
                                                 parentConnectionPointOpeningAngle_track[1, :].reshape(nLinks_track, 1), \
                                                 parentIsPOIClosestToNu_track[1, :].reshape(nLinks_track, 1), \
                                                 childIsPOIClosestToNu_track[1, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentNuVertexSep_track[2, :].reshape(nLinks_track, 1), \
                                                 childNuVertexSep_track[2, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNHits_track[2, :].reshape(nLinks_track, 1), \
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
                                                 parentConnectionPointOpeningAngle_track[2, :].reshape(nLinks_track, 1), \
                                                 parentIsPOIClosestToNu_track[2, :].reshape(nLinks_track, 1), \
                                                 childIsPOIClosestToNu_track[2, :].reshape(nLinks_track, 1)), axis=1), \
                                 np.concatenate((parentNuVertexSep_track[3, :].reshape(nLinks_track, 1), \
                                                 childNuVertexSep_track[3, :].reshape(nLinks_track, 1), \
                                                 parentEndRegionNHits_track[3, :].reshape(nLinks_track, 1), \
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
                                                 parentConnectionPointOpeningAngle_track[3, :].reshape(nLinks_track, 1), \
                                                 parentIsPOIClosestToNu_track[3, :].reshape(nLinks_track, 1), \
                                                 childIsPOIClosestToNu_track[3, :].reshape(nLinks_track, 1)), axis=1)), axis=1)

    coc0_shower = np.concatenate((np.concatenate((parentNuVertexSep_shower[0, :].reshape(nLinks_shower, 1), \
                                                 childNuVertexSep_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionNHits_shower[0, :].reshape(nLinks_shower, 1), \
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
                                                 parentConnectionPointOpeningAngle_shower[0, :].reshape(nLinks_shower, 1), \
                                                 parentIsPOIClosestToNu_shower[0, :].reshape(nLinks_shower, 1), \
                                                 childIsPOIClosestToNu_shower[0, :].reshape(nLinks_shower, 1)), axis=1), \
                                 np.concatenate((parentNuVertexSep_shower[1, :].reshape(nLinks_shower, 1), \
                                                 childNuVertexSep_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentEndRegionNHits_shower[1, :].reshape(nLinks_shower, 1), \
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
                                                 parentConnectionPointOpeningAngle_shower[1, :].reshape(nLinks_shower, 1), \
                                                 parentIsPOIClosestToNu_shower[1, :].reshape(nLinks_shower, 1), \
                                                 childIsPOIClosestToNu_shower[1, :].reshape(nLinks_shower, 1)), axis=1)), axis=1)
    
    # concatenate variable_single and orientations
    variables_track = np.concatenate((parentTrackScore_track.reshape(nLinks_track, 1), \
                                      childTrackScore_track.reshape(nLinks_track, 1), \
                                      parentNSpacepoints_track.reshape(nLinks_track, 1), \
                                      childNSpacepoints_track.reshape(nLinks_track, 1), \
                                      separation3D_track.reshape(nLinks_track, 1), \
                                      pidLinkType_track.reshape(nLinks_track, 1), \
                                      coc0_track), axis=1)

    variables_shower = np.concatenate((parentTrackScore_shower.reshape(nLinks_shower, 1), \
                                      childTrackScore_shower.reshape(nLinks_shower, 1), \
                                      parentNSpacepoints_shower.reshape(nLinks_shower, 1), \
                                      childNSpacepoints_shower.reshape(nLinks_shower, 1), \
                                      separation3D_shower.reshape(nLinks_shower, 1), \
                                      pidLinkType_shower.reshape(nLinks_shower, 1), \
                                      coc0_shower), axis=1)
    
    return parentPFPIndex_track, childPFPIndex_track, variables_track, y_track, trueParentChildLink_track, trainingCutSep_track, trainingCutL_track, trainingCutT_track, \
parentPFPIndex_shower, childPFPIndex_shower, variables_shower, y_shower, trueParentChildLink_shower, trainingCutSep_shower, trainingCutL_shower, trainingCutT_shower

############################################################################################################################################
############################################################################################################################################
        
def CalculateTrainingCuts(childStartX, childStartY, childStartZ, childStartDX, childStartDY, childStartDZ, parentEndX, parentEndY, parentEndZ, parentEndDX, parentEndDY, parentEndDZ) :
    childStartPos = np.array([childStartX, childStartY, childStartZ])
    childStartDir = np.array([childStartDX, childStartDY, childStartDZ]) * -1.0 # Need to turn it around
    parentEndPos = np.array([parentEndX, parentEndY, parentEndZ])
    parentEndDir = np.array([parentEndDX, parentEndDY, parentEndDZ])
    
    smallestT = 999999999999999
    connectionPoint = np.array([-999.0, -999.0, -999.0])
    found = False
    
    extrapolatedPoint = childStartPos
    
    while (IsInFV(extrapolatedPoint)) :
        
        extrapolatedPoint = extrapolatedPoint + (childStartDir * 1.0)
        parentDir_t = np.linalg.norm(np.cross(parentEndDir, (extrapolatedPoint - parentEndPos)))
        
        if (parentDir_t < smallestT) : 
            smallestT = parentDir_t
            connectionPoint = extrapolatedPoint
            found = True
            
    childDir_t = np.linalg.norm(childStartPos - connectionPoint) if found else -999.0
    parentDir_l = np.dot(parentEndDir, (connectionPoint - parentEndPos)) if found else -999.0

    return parentDir_l, childDir_t        
        
############################################################################################################################################
############################################################################################################################################
        
def IsInFV(position_np) :
    
    minX = -360.0
    maxX = 360.0
    minY = -600.0 
    maxY = 600.0
    minZ = 0.0
    maxZ = 1394.0

    if ((position_np[0] < minX) or (position_np[0] > maxX)) :
        return False;

    if ((position_np[1] < minY) or (position_np[1] > maxY)) :
        return False;

    if ((position_np[2] < minZ) or (position_np[2] > maxZ)) :
        return False
    
    return True
        

    