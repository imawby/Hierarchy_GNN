import numpy as np
import uproot
import math

import copy

from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

###################################
###################################

primaryNSpacepoints_min = 0.0
primaryNSpacepoints_max = 2000.0
primaryNuVertexSeparation_min = -50.0
primaryNuVertexSeparation_max = 500.0
primaryStartRegionNHits_min = -10.0
primaryStartRegionNHits_max = 100.0
primaryStartRegionNParticles_min = -1.0
primaryStartRegionNParticles_max = 8.0
primaryDCA_min = -60.0
primaryDCA_max = 600.0
primaryConnectionExtrapDistance_min = -700.0
primaryConnectionExtrapDistance_max = 500.0
primaryClosestParentL_min = -150.0
primaryClosestParentL_max = 150.0
primaryClosestParentT_min = -30.0
primaryClosestParentT_max = 300.0
primaryOpeningAngle_min = -0.5
primaryOpeningAngle_max = 3.14

############################################################################################################################################
############################################################################################################################################

def normaliseXAxis(variable, minLimit, maxLimit) :

    interval = math.fabs(minLimit) + math.fabs(maxLimit)
    variable[variable < minLimit] = minLimit
    variable[variable > maxLimit] = maxLimit
    variable /= interval
    
############################################################################################################################################
############################################################################################################################################    
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

############################################################################################################################################
############################################################################################################################################

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    

############################################################################################################################################
############################################################################################################################################    

def readTreeGroupLinks_track(fileNames) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    primaryTrackScore = []
    primaryNSpacepoints = []
    primaryNuVertexSeparation = [[], []]
    primaryStartRegionNHits = [[], []]
    primaryStartRegionNParticles = [[], []]
    primaryDCA = [[], []]
    primaryConnectionExtrapDistance = [[], []]
    primaryIsPOIClosestToNu = [[], []]
    primaryClosestParentL = [[], []]
    primaryClosestParentT = [[], []]
    primaryOpeningAngle = [[], []]
    
    # Training cut 
    trainingCutDCA = [] # only fill for correct orientation
    hasNeutronParent = []
    
    # Truth
    isTruePrimaryLink = []
    isLinkOrientationCorrect = []
    y = []

        
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            if ((iEvent % 100) == 0) :
                print('iEvent:', str(iEvent) + '/' + str(nEvents))            
            
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
            primaryTrackShowerScore_file = np.array(branches['PrimaryTrackScore'][iEvent])
            primaryTrack_mask = (primaryTrackShowerScore_file > 0.5)
                
            if (np.count_nonzero(primaryTrack_mask) == 0) :
                continue
            ##################################################################    
            primaryTrackShowerScore_file = primaryTrackShowerScore_file[primaryTrack_mask]
            primaryPFPIndex_file = np.array(branches['PrimaryPFPIndex'][iEvent][primaryTrack_mask])    
            primaryNSpacepoints_file = np.array(branches['PrimaryNSpacepoints'][iEvent][primaryTrack_mask])
            primaryNuVertexSeparation_file = np.array(branches['PrimaryNuVertexSeparation'][iEvent][primaryTrack_mask])
            primaryStartRegionNHits_file = np.array(branches['PrimaryStartRegionNHits'][iEvent][primaryTrack_mask])            
            primaryStartRegionNParticles_file = np.array(branches['PrimaryStartRegionNParticles'][iEvent][primaryTrack_mask])            
            primaryDCA_file = np.array(branches['PrimaryDCA'][iEvent][primaryTrack_mask])            
            primaryConnectionExtrapDistance_file = np.array(branches['PrimaryConnectionExtrapDistance'][iEvent][primaryTrack_mask])
            primaryIsPOIClosestToNu_file = np.array(branches['PrimaryIsPOIClosestToNu'][iEvent][primaryTrack_mask])
            primaryStartX_file = np.array(branches['PrimaryStartX'][iEvent][primaryTrack_mask])
            primaryStartY_file = np.array(branches['PrimaryStartY'][iEvent][primaryTrack_mask])
            primaryStartZ_file = np.array(branches['PrimaryStartZ'][iEvent][primaryTrack_mask])
            primaryStartDX_file = np.array(branches['PrimaryStartDX'][iEvent][primaryTrack_mask])
            primaryStartDY_file = np.array(branches['PrimaryStartDY'][iEvent][primaryTrack_mask])
            primaryStartDZ_file = np.array(branches['PrimaryStartDZ'][iEvent][primaryTrack_mask])
            # True
            isTruePrimaryLink_file = np.array(branches['TruePrimaryLink'][iEvent][primaryTrack_mask])
            trueParentPDG_file = np.array(branches['RecoPFPTrueParentPDG'][iEvent]) # we really dont want to take a mask here!
            isLinkOrientationCorrect_file = np.array(branches['IsPrimaryLinkOrientationCorrect'][iEvent][primaryTrack_mask])            
            
            ####################################
            # Now loop over loops to group them.
            ####################################
            currentPFP = -1
            linksMadeCounter = 0
                
            this_y = [0, 0]
            this_isLinkOrientationCorrect = [0, 0]
            order = [0, 1]

            for iLink in range(0, primaryPFPIndex_file.shape[0]) :
                
                # If we have moved onto a new group...
                if (currentPFP != primaryPFPIndex_file[iLink]) :
                    
                    if len(np.where(primaryPFPIndex_file == primaryPFPIndex_file[iLink])[0]) != 2 :
                        continue

                    # set the common vars
                    primaryNSpacepoints.append(primaryNSpacepoints_file[iLink])   
                    isTruePrimaryLink.append(isTruePrimaryLink_file[iLink])
                    hasNeutronParent.append(trueParentPDG_file[primaryPFPIndex_file[iLink]] == 2112)
                    currentPFP = primaryPFPIndex_file[iLink]

                # Set truth
                if (isTruePrimaryLink_file[iLink] and isLinkOrientationCorrect_file[iLink]) :
                    this_y[order[linksMadeCounter]] = 1 
                elif (isTruePrimaryLink_file[iLink] and (not isLinkOrientationCorrect_file[iLink])) :
                    this_y[order[linksMadeCounter]] = 2

                this_isLinkOrientationCorrect[order[linksMadeCounter]] = isLinkOrientationCorrect_file[iLink]

                # set the link information
                primaryNuVertexSeparation[order[linksMadeCounter]].append(primaryNuVertexSeparation_file[iLink])
                primaryStartRegionNHits[order[linksMadeCounter]].append(primaryStartRegionNHits_file[iLink])
                primaryStartRegionNParticles[order[linksMadeCounter]].append(primaryStartRegionNParticles_file[iLink])
                primaryDCA[order[linksMadeCounter]].append(primaryDCA_file[iLink])
                primaryConnectionExtrapDistance[order[linksMadeCounter]].append(primaryConnectionExtrapDistance_file[iLink])
                primaryIsPOIClosestToNu[order[linksMadeCounter]].append(1 if primaryIsPOIClosestToNu_file[iLink] else 0)
                    
                # Get opening angle
                primaryStartVector = np.array([primaryStartX_file[iLink], primaryStartY_file[iLink], primaryStartZ_file[iLink]])
                displacementVector = primaryStartVector - np.array([branches['RecoNuVtxX'][iEvent], branches['RecoNuVtxY'][iEvent], branches['RecoNuVtxZ'][iEvent]])
                primaryOpeningAngle[order[linksMadeCounter]].append(angle_between(displacementVector, primaryStartVector))
                
                # Look for parent L and T
                this_primaryClosestParentL = -999.0
                this_primaryClosestParentT = -999.0
                
                parentPFPIndex_file = branches['ParentPFPIndex'][iEvent]
                childPFPIndex_file = branches['ChildPFPIndex'][iEvent]
                isChildPOIClosestToNu_file = branches['IsChildPOIClosestToNu'][iEvent]
                isParentPOIClosestToNu_file = branches['IsParentPOIClosestToNu'][iEvent]
                parentNuVertexSeparation_file = branches['ParentNuVertexSeparation'][iEvent]
                trainingCutL_file = branches['TrainingCutL'][iEvent]
                trainingCutT_file = branches['TrainingCutT'][iEvent]
                
                higherNetworkLinkIndex = np.where(childPFPIndex_file == primaryPFPIndex_file[iLink])[0]
                
                if (len(higherNetworkLinkIndex) != 0) :
                    this_parentDisplacement = 0
                    this_parentL = 0
                    this_parentT = 0
                    count = 0
                    
                    for higherNetworkLink in higherNetworkLinkIndex :
                        
                        count = count + 1
                        
                        if (primaryIsPOIClosestToNu_file[iLink] == isChildPOIClosestToNu_file[higherNetworkLink]) : 

                            if (isParentPOIClosestToNu_file[higherNetworkLink]) :
                                this_parentDisplacement = parentNuVertexSeparation_file[higherNetworkLink]                            
                            else :
                                this_parentL = trainingCutL_file[higherNetworkLink]                            
                                this_parentT = trainingCutT_file[higherNetworkLink]

                        # We know that the child is a track, and parent has to be a track too
                        if (count == 4) :
                            count = 0
                            
                            if ((this_parentDisplacement < primaryNuVertexSeparation_file[iLink]) and (abs(this_parentL) < abs(this_primaryClosestParentL))):
                                this_primaryClosestParentL = this_parentL
                                this_primaryClosestParentT = this_parentT

                primaryClosestParentL[order[linksMadeCounter]].append(this_primaryClosestParentL)
                primaryClosestParentT[order[linksMadeCounter]].append(this_primaryClosestParentT)
                        
                # Add in training cuts 
                if (isLinkOrientationCorrect_file[iLink]) :
                    trainingCutDCA.append(primaryDCA_file[iLink])
                                
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
    primaryTrackScore = np.array(primaryTrackScore, dtype='float64')
    primaryNSpacepoints = np.array(primaryNSpacepoints, dtype='float64')
    primaryNuVertexSeparation = np.array(primaryNuVertexSeparation, dtype='float64')
    primaryStartRegionNHits = np.array(primaryStartRegionNHits, dtype='float64')
    primaryStartRegionNParticles = np.array(primaryStartRegionNParticles, dtype='float64')
    primaryDCA = np.array(primaryDCA, dtype='float64')
    primaryConnectionExtrapDistance = np.array(primaryConnectionExtrapDistance, dtype='float64')
    primaryIsPOIClosestToNu = np.array(primaryIsPOIClosestToNu, dtype='int')
    primaryClosestParentL = np.array(primaryClosestParentL, dtype='float64')
    primaryClosestParentT = np.array(primaryClosestParentT, dtype='float64')
    primaryOpeningAngle = np.array(primaryOpeningAngle, dtype='float64')

    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
    hasNeutronParent = np.array(hasNeutronParent)
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect)
    y = np.array(y)
        
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = isTruePrimaryLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')        
          
    ###################################
    # Normalise variables
    ###################################        
    normaliseXAxis(primaryNSpacepoints, primaryNSpacepoints_min, primaryNSpacepoints_max)    
    normaliseXAxis(primaryNuVertexSeparation, primaryNuVertexSeparation_min, primaryNuVertexSeparation_max)    
    normaliseXAxis(primaryStartRegionNHits, primaryStartRegionNHits_min, primaryStartRegionNHits_max)    
    normaliseXAxis(primaryStartRegionNParticles, primaryStartRegionNParticles_min, primaryStartRegionNParticles_max)    
    normaliseXAxis(primaryDCA, primaryDCA_min, primaryDCA_max)
    normaliseXAxis(primaryConnectionExtrapDistance, primaryConnectionExtrapDistance_min, primaryConnectionExtrapDistance_max)
    normaliseXAxis(primaryClosestParentL, primaryClosestParentL_min, primaryClosestParentL_max) 
    normaliseXAxis(primaryClosestParentT, primaryClosestParentT_min, primaryClosestParentT_max) 
    normaliseXAxis(primaryOpeningAngle, primaryOpeningAngle_min, primaryOpeningAngle_max) 
        
    ###################################
    # Concatenate
    ###################################
    primaryClosestParentL = np.array(primaryClosestParentL, dtype='float64')
    primaryClosestParentT = np.array(primaryClosestParentT, dtype='float64')
    primaryOpeningAngle = np.array(primaryOpeningAngle, dtype='float64')    
    
    
    coc0 = np.concatenate((np.concatenate((primaryNuVertexSeparation[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[0, :].reshape(nLinks, 1), \
                                           primaryDCA[0, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[0, :].reshape(nLinks, 1), \
                                           primaryIsPOIClosestToNu[0, :].reshape(nLinks, 1), \
                                           primaryClosestParentL[0, :].reshape(nLinks, 1), \
                                           primaryClosestParentT[0, :].reshape(nLinks, 1), \
                                           primaryOpeningAngle[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((primaryNuVertexSeparation[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[1, :].reshape(nLinks, 1), \
                                           primaryDCA[1, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[1, :].reshape(nLinks, 1), \
                                           primaryIsPOIClosestToNu[1, :].reshape(nLinks, 1), \
                                           primaryClosestParentL[1, :].reshape(nLinks, 1), \
                                           primaryClosestParentT[1, :].reshape(nLinks, 1), \
                                           primaryOpeningAngle[1, :].reshape(nLinks, 1)), axis=1)), axis=1)      
    
    # concatenate variable_single and orientations
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                                coc0), axis=1)

    
    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect, trainingCutDCA, hasNeutronParent
    
############################################################################################################################################
############################################################################################################################################        

def readTreeGroupLinks_shower(fileNames) :
        
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    primaryNSpacepoints = []
    primaryNuVertexSeparation = []
    primaryStartRegionNHits = []
    primaryStartRegionNParticles = []
    primaryDCA = []
    primaryConnectionExtrapDistance = []
    primaryIsPOIClosestToNu = []
    primaryClosestParentL = []
    primaryClosestParentT = []
    primaryOpeningAngle = []
    # Training cut 
    trainingCutDCA = [] # only fill for correct orientation
    # Truth
    isTruePrimaryLink = []
    hasNeutronParent = []
    isLinkOrientationCorrect = []
    y = []

        
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ccnuselection/ccnusel']
        branches = tree.arrays()
        
        nEvents = len(branches)
        
        for iEvent in range(nEvents) :
            
            if ((iEvent % 100) == 0) :
                print('iEvent:', str(iEvent) + '/' + str(nEvents))
            
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
            # DEFINE THEM ALL HERE - apply shower mask
            ##################################################################
            primaryTrackShowerScore_file = np.array(branches['PrimaryTrackScore'][iEvent])
            primaryShower_mask = (primaryTrackShowerScore_file < 0.5)
                
            if (np.count_nonzero(primaryShower_mask) == 0) :
                continue
                
            ##################################################################    
            primaryPFPIndex_file = np.array(branches['PrimaryPFPIndex'][iEvent][primaryShower_mask]) 
            primaryNuVertexSeparation_file = np.array(branches['PrimaryNuVertexSeparation'][iEvent][primaryShower_mask])
            isTruePrimaryLink_file = branches['TruePrimaryLink'][iEvent][primaryShower_mask]
            trueParentPDG_file = np.array(branches['RecoPFPTrueParentPDG'][iEvent]) # we really dont want to take a mask here!
            isLinkOrientationCorrect_file = branches['IsPrimaryLinkOrientationCorrect'][iEvent][primaryShower_mask]
            primaryStartX_file = np.array(branches['PrimaryStartX'][iEvent][primaryShower_mask])
            primaryStartY_file = np.array(branches['PrimaryStartY'][iEvent][primaryShower_mask])
            primaryStartZ_file = np.array(branches['PrimaryStartZ'][iEvent][primaryShower_mask])
            primaryStartDX_file = np.array(branches['PrimaryStartDX'][iEvent][primaryShower_mask])
            primaryStartDY_file = np.array(branches['PrimaryStartDY'][iEvent][primaryShower_mask])
            primaryStartDZ_file = np.array(branches['PrimaryStartDZ'][iEvent][primaryShower_mask]) 
            # set the link information
            primaryNSpacepoints.extend(branches['PrimaryNSpacepoints'][iEvent][primaryShower_mask])
            primaryNuVertexSeparation.extend(branches['PrimaryNuVertexSeparation'][iEvent][primaryShower_mask])
            primaryStartRegionNHits.extend(branches['PrimaryStartRegionNHits'][iEvent][primaryShower_mask])            
            primaryStartRegionNParticles.extend(branches['PrimaryStartRegionNParticles'][iEvent][primaryShower_mask])            
            primaryDCA.extend(branches['PrimaryDCA'][iEvent][primaryShower_mask])            
            primaryConnectionExtrapDistance.extend(branches['PrimaryConnectionExtrapDistance'][iEvent][primaryShower_mask])
            primaryIsPOIClosestToNu.extend(branches['PrimaryIsPOIClosestToNu'][iEvent][primaryShower_mask])           
            # True
            isTruePrimaryLink.extend(isTruePrimaryLink_file)
            isLinkOrientationCorrect.extend(isLinkOrientationCorrect_file)
            # Add in training cuts 
            trainingCutDCA.extend(branches['PrimaryDCA'][iEvent][primaryShower_mask])
            
            # Going to have to loop over the links to work these things out :( 
            for iLink in range(0, primaryPFPIndex_file.shape[0]) :
                
                # hasNeutronParent
                hasNeutronParent.append(trueParentPDG_file[primaryPFPIndex_file[iLink]] == 2112)
                
                # Get opening angle
                primaryStartVector = np.array([primaryStartX_file[iLink], primaryStartY_file[iLink], primaryStartZ_file[iLink]])
                displacementVector = primaryStartVector - np.array([branches['RecoNuVtxX'][iEvent], branches['RecoNuVtxY'][iEvent], branches['RecoNuVtxZ'][iEvent]])
                primaryOpeningAngle.append(angle_between(displacementVector, primaryStartVector))
                
                # Look for parent L and T
                this_primaryClosestParentL = -999.0
                this_primaryClosestParentT = -999.0
                
                parentPFPIndex_file = branches['ParentPFPIndex'][iEvent]
                childPFPIndex_file = branches['ChildPFPIndex'][iEvent]
                isChildPOIClosestToNu_file = branches['IsChildPOIClosestToNu'][iEvent]
                isParentPOIClosestToNu_file = branches['IsParentPOIClosestToNu'][iEvent]
                parentNuVertexSeparation_file = branches['ParentNuVertexSeparation'][iEvent]
                trainingCutL_file = branches['TrainingCutL'][iEvent]
                trainingCutT_file = branches['TrainingCutT'][iEvent]
                
                higherNetworkLinkIndex = np.where(childPFPIndex_file == primaryPFPIndex_file[iLink])[0]
                
                if (len(higherNetworkLinkIndex) != 0) :
                    this_parentDisplacement = 0
                    this_parentL = 0
                    this_parentT = 0
                    count = 0
                    
                    for higherNetworkLink in higherNetworkLinkIndex :
                        
                        count = count + 1

                        if (isParentPOIClosestToNu_file[higherNetworkLink]) :
                            this_parentDisplacement = parentNuVertexSeparation_file[higherNetworkLink]                            
                        else :
                            this_parentL = trainingCutL_file[higherNetworkLink]                            
                            this_parentT = trainingCutT_file[higherNetworkLink]
                        
                        # We know that the child is a shower, and parent has to be a track
                        if (count == 2) :
                            count = 0
                            
                            if ((this_parentDisplacement < primaryNuVertexSeparation_file[iLink]) and (abs(this_parentL) < abs(this_primaryClosestParentL))):
                                this_primaryClosestParentL = this_parentL
                                this_primaryClosestParentT = this_parentT

                primaryClosestParentL.append(this_primaryClosestParentL)
                primaryClosestParentT.append(this_primaryClosestParentT)
        
            this_y = np.zeros(np.array(isTruePrimaryLink_file).shape)
            this_y[np.logical_and(isTruePrimaryLink_file, isLinkOrientationCorrect_file)] = 1
            this_y[np.logical_and(isTruePrimaryLink_file, np.logical_not(isLinkOrientationCorrect_file))] = 2
            y.extend(this_y)

            
    ###################################
    # Now turn things into numpy arrays
    ###################################
    primaryNSpacepoints = np.array(primaryNSpacepoints, dtype='float64')
    primaryNuVertexSeparation = np.array(primaryNuVertexSeparation, dtype='float64')
    primaryStartRegionNHits = np.array(primaryStartRegionNHits, dtype='float64')
    primaryStartRegionNParticles = np.array(primaryStartRegionNParticles, dtype='float64')
    primaryDCA = np.array(primaryDCA, dtype='float64')
    primaryConnectionExtrapDistance = np.array(primaryConnectionExtrapDistance, dtype='float64')
    primaryIsPOIClosestToNu = np.array(primaryIsPOIClosestToNu, dtype='int')
    primaryClosestParentL = np.array(primaryClosestParentL, dtype='float64')
    primaryClosestParentT = np.array(primaryClosestParentT, dtype='float64')
    primaryOpeningAngle = np.array(primaryOpeningAngle, dtype='float64')
    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
    hasNeutronParent = np.array(hasNeutronParent)
    isLinkOrientationCorrect = np.array(isLinkOrientationCorrect)
    y = np.array(y)
        
    ###################################
    # How many links do we have?
    ###################################        
    nLinks = isTruePrimaryLink.shape[0]   
    print('We have ', str(nLinks), ' to train on!')        
          
    ###################################
    # Normalise variables
    ###################################        
    normaliseXAxis(primaryNSpacepoints, primaryNSpacepoints_min, primaryNSpacepoints_max)    
    normaliseXAxis(primaryNuVertexSeparation, primaryNuVertexSeparation_min, primaryNuVertexSeparation_max)    
    normaliseXAxis(primaryStartRegionNHits, primaryStartRegionNHits_min, primaryStartRegionNHits_max)    
    normaliseXAxis(primaryStartRegionNParticles, primaryStartRegionNParticles_min, primaryStartRegionNParticles_max)    
    normaliseXAxis(primaryDCA, primaryDCA_min, primaryDCA_max)
    normaliseXAxis(primaryConnectionExtrapDistance, primaryConnectionExtrapDistance_min, primaryConnectionExtrapDistance_max) 
    normaliseXAxis(primaryClosestParentL, primaryClosestParentL_min, primaryClosestParentL_max) 
    normaliseXAxis(primaryClosestParentT, primaryClosestParentT_min, primaryClosestParentT_max) 
    normaliseXAxis(primaryOpeningAngle, primaryOpeningAngle_min, primaryOpeningAngle_max)     
        
    ###################################
    # Concatenate
    ###################################
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                           primaryNuVertexSeparation.reshape(nLinks, 1), \
                           primaryStartRegionNHits.reshape(nLinks, 1), \
                           primaryStartRegionNParticles.reshape(nLinks, 1), \
                           primaryDCA.reshape(nLinks, 1), \
                           primaryConnectionExtrapDistance.reshape(nLinks, 1), \
                           primaryIsPOIClosestToNu.reshape(nLinks, 1), \
                           primaryClosestParentL.reshape(nLinks, 1), \
                           primaryClosestParentT.reshape(nLinks, 1), \
                           primaryOpeningAngle.reshape(nLinks, 1)), axis=1)    

    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect, trainingCutDCA, hasNeutronParent
        
        
############################################################################################################################################
############################################################################################################################################        

def readEvent(eventDict, higherTierDict) :
         
    ###################################
    # nu-track links
    ###################################
    # Link variables
    primaryTrackScore_track = []
    primaryNSpacepoints_track = []
    primaryNuVertexSeparation_track = [[], []]
    primaryStartRegionNHits_track = [[], []]
    primaryStartRegionNParticles_track = [[], []]
    primaryDCA_track = [[], []]
    primaryConnectionExtrapDistance_track = [[], []]
    primaryIsPOIClosestToNu_track = [[], []]
    primaryClosestParentL_track = [[], []]
    primaryClosestParentT_track = [[], []]
    # ID
    primaryPFPIndex_track = []
    # training
    trainingDCA_track_temp = [[], []]
    # Truth
    isTruePrimaryLink_track = []
    y_track = []
    ###################################
    # nu-shower links
    ###################################
    # Link variables
    primaryTrackScore_shower = []
    primaryNSpacepoints_shower = []
    primaryNuVertexSeparation_shower = []
    primaryStartRegionNHits_shower = []
    primaryStartRegionNParticles_shower = []
    primaryDCA_shower = []
    primaryConnectionExtrapDistance_shower = []
    primaryIsPOIClosestToNu_shower = []
    primaryClosestParentL_shower = []
    primaryClosestParentT_shower = []    
    # ID
    primaryPFPIndex_shower = []
    # training
    trainingDCA_shower = []
    # Truth
    isTruePrimaryLink_shower = []

    ####################################
    # Now loop over loops to group them.
    ####################################
    currentPFP = -1
    linksMadeCounter = 0                
    this_y_track = [0, 0]
        
    for iLink in range(eventDict['primaryPFPIndex'].shape[0]) :
                
        primaryTrackShowerScore_event = eventDict['primaryTrackShowerScore'][iLink]
        isTrack = (primaryTrackShowerScore_event > 0.5)
            
        # If we have moved onto a new group...
        if (currentPFP != eventDict['primaryPFPIndex'][iLink]) :
            
            if isTrack and len(np.where(eventDict['primaryPFPIndex'] == eventDict['primaryPFPIndex'][iLink])[0]) != 2 :
                continue
            
            # set the common vars
            primaryNSpacepoints_track.append(eventDict['primaryNSpacepoints'][iLink]) if isTrack else primaryNSpacepoints_shower.append(eventDict['primaryNSpacepoints'][iLink])
            primaryPFPIndex_track.append(eventDict['primaryPFPIndex'][iLink]) if isTrack else primaryPFPIndex_shower.append(eventDict['primaryPFPIndex'][iLink])
            isTruePrimaryLink_track.append(eventDict['isTruePrimaryLink'][iLink]) if isTrack else isTruePrimaryLink_shower.append(eventDict['isTruePrimaryLink'][iLink])
            
            currentPFP = eventDict['primaryPFPIndex'][iLink]

        # Set truth - only need to do this for nu-track
        if (isTrack) :
            if (eventDict['isTruePrimaryLink'][iLink] and eventDict['isLinkOrientationCorrect'][iLink]) :
                this_y_track[linksMadeCounter] = 1 
            elif (eventDict['isTruePrimaryLink'][iLink] and (not eventDict['isLinkOrientationCorrect'][iLink])) :
                this_y_track[linksMadeCounter] = 2

        # set the link information
        primaryNuVertexSeparation_track[linksMadeCounter].append(eventDict['primaryNuVertexSeparation'][iLink]) if isTrack else \
            primaryNuVertexSeparation_shower.append(eventDict['primaryNuVertexSeparation'][iLink])
        primaryStartRegionNHits_track[linksMadeCounter].append(eventDict['primaryStartRegionNHits'][iLink]) if isTrack else \
            primaryStartRegionNHits_shower.append(eventDict['primaryStartRegionNHits'][iLink])
        primaryStartRegionNParticles_track[linksMadeCounter].append(eventDict['primaryStartRegionNParticles'][iLink]) if isTrack else \
            primaryStartRegionNParticles_shower.append(eventDict['primaryStartRegionNParticles'][iLink])
        primaryDCA_track[linksMadeCounter].append(eventDict['primaryDCA'][iLink]) if isTrack else \
            primaryDCA_shower.append(eventDict['primaryDCA'][iLink])
        primaryConnectionExtrapDistance_track[linksMadeCounter].append(eventDict['primaryConnectionExtrapDistance'][iLink]) if isTrack else \
            primaryConnectionExtrapDistance_shower.append(eventDict['primaryConnectionExtrapDistance'][iLink]) 
        primaryIsPOIClosestToNu_track[linksMadeCounter].append(1 if eventDict['primaryIsPOIClosestToNu'][iLink] else 0) if isTrack else \
            primaryIsPOIClosestToNu_shower.append(1 if eventDict['primaryIsPOIClosestToNu'][iLink] else 0)
                
        # Look for parent L and T
        this_primaryClosestParentL = -999.0
        this_primaryClosestParentT = -999.0
                
        print('NEW')    
            
        higherNetworkLinkIndex = np.where(higherTierDict['childPFPIndex'] == eventDict['primaryPFPIndex'][iLink])[0]
                
        if (len(higherNetworkLinkIndex) != 0) :
            this_parentDisplacement = 0
            this_parentL = 0
            this_parentT = 0
            count = 0
                    
            for higherNetworkLink in higherNetworkLinkIndex :

                count = count + 1

                if ((not isTrack) or (isTrack and (eventDict['primaryIsPOIClosestToNu'][iLink] == higherTierDict['isChildPOIClosestToNu'][higherNetworkLink]))) : 

                    if (higherTierDict['isParentPOIClosestToNu'][higherNetworkLink]) :
                        this_parentDisplacement = higherTierDict['parentNuVertexSeparation'][higherNetworkLink]                            
                    else :
                        this_parentL = higherTierDict['trainingCutL'][higherNetworkLink]                            
                        this_parentT = higherTierDict['trainingCutT'][higherNetworkLink]

                # We know that the child is a shower, and parent has to be a track
                if ((isTrack and count == 4) or (not isTrack and count == 2)) :
                    count = 0

#                     print('this_parentL: ', this_parentL)
#                     print('eventDict[primaryNuVertexSeparation][iLink]: ', eventDict['primaryNuVertexSeparation'][iLink])
#                     print('this_parentDisplacement: ', this_parentDisplacement)
                    
                    if ((this_parentDisplacement < eventDict['primaryNuVertexSeparation'][iLink]) and (abs(this_parentL) < abs(this_primaryClosestParentL))):
                                                
                        this_primaryClosestParentL = this_parentL
                        this_primaryClosestParentT = this_parentT
                        
#                     print('this_primaryClosestParentL:', this_primaryClosestParentL)

        primaryClosestParentL_track[linksMadeCounter].append(this_primaryClosestParentL) if isTrack else \
            primaryClosestParentL_shower.append(this_primaryClosestParentL)                      
        primaryClosestParentT_track[linksMadeCounter].append(this_primaryClosestParentT) if isTrack else \
            primaryClosestParentT_shower.append(this_primaryClosestParentT)    
        # training cut
        trainingDCA_track_temp[linksMadeCounter].append(eventDict['primaryDCA'][iLink]) if isTrack else \
            trainingDCA_shower.append(eventDict['primaryDCA'][iLink])
        
        linksMadeCounter = linksMadeCounter + 1

        if (isTrack and (linksMadeCounter == 2)) :
            y_track.append(this_y_track)                       
            linksMadeCounter = 0
            this_y_track = [0, 0]
            
        if (not isTrack) :
            linksMadeCounter = 0
            
    #########################################################################
    # Want trainingDCA_track to have one number per particle-particle match
    #########################################################################            
    trainingDCA_track = [min(trainingDCA_track_temp[0][i], trainingDCA_track_temp[1][i]) for i in range(len(trainingDCA_track_temp[0]))]
            
    ###################################
    # Now turn things into numpy arrays
    ###################################
    primaryNSpacepoints_track = np.array(primaryNSpacepoints_track, dtype='float64')
    primaryNuVertexSeparation_track = np.array(primaryNuVertexSeparation_track, dtype='float64')
    primaryStartRegionNHits_track = np.array(primaryStartRegionNHits_track, dtype='float64')
    primaryStartRegionNParticles_track = np.array(primaryStartRegionNParticles_track, dtype='float64')
    primaryDCA_track = np.array(primaryDCA_track, dtype='float64')
    primaryConnectionExtrapDistance_track = np.array(primaryConnectionExtrapDistance_track, dtype='float64')
    primaryIsPOIClosestToNu_track = np.array(primaryIsPOIClosestToNu_track, dtype='int')
    primaryClosestParentL_track = np.array(primaryClosestParentL_track, dtype='float64')
    primaryClosestParentT_track = np.array(primaryClosestParentT_track, dtype='float64')
    primaryPFPIndex_track = np.array(primaryPFPIndex_track, dtype='int')
    isTruePrimaryLink_track = np.array(isTruePrimaryLink_track, dtype='int')
    y_track = np.array(y_track, dtype='int')
    trainingDCA_track = np.array(trainingDCA_track, dtype='float64')
            
    primaryNSpacepoints_shower = np.array(primaryNSpacepoints_shower, dtype='float64')
    primaryNuVertexSeparation_shower = np.array(primaryNuVertexSeparation_shower, dtype='float64')
    primaryStartRegionNHits_shower = np.array(primaryStartRegionNHits_shower, dtype='float64')
    primaryStartRegionNParticles_shower = np.array(primaryStartRegionNParticles_shower, dtype='float64')
    primaryDCA_shower = np.array(primaryDCA_shower, dtype='float64')
    primaryConnectionExtrapDistance_shower = np.array(primaryConnectionExtrapDistance_shower, dtype='float64')
    primaryIsPOIClosestToNu_shower = np.array(primaryIsPOIClosestToNu_shower, dtype='int')
    primaryClosestParentL_shower = np.array(primaryClosestParentL_shower, dtype='float64')
    primaryClosestParentT_shower = np.array(primaryClosestParentT_shower, dtype='float64')    
    primaryPFPIndex_shower = np.array(primaryPFPIndex_shower, dtype='int')
    isTruePrimaryLink_shower = np.array(isTruePrimaryLink_shower, dtype='int')
    trainingDCA_shower = np.array(trainingDCA_shower, dtype='float64')

    ###################################
    # Need to normalise!
    ###################################    
    normaliseXAxis(primaryNSpacepoints_track, primaryNSpacepoints_min, primaryNSpacepoints_max)    
    normaliseXAxis(primaryNuVertexSeparation_track, primaryNuVertexSeparation_min, primaryNuVertexSeparation_max)    
    normaliseXAxis(primaryStartRegionNHits_track, primaryStartRegionNHits_min, primaryStartRegionNHits_max)    
    normaliseXAxis(primaryStartRegionNParticles_track, primaryStartRegionNParticles_min, primaryStartRegionNParticles_max)    
    normaliseXAxis(primaryDCA_track, primaryDCA_min, primaryDCA_max)
    normaliseXAxis(primaryConnectionExtrapDistance_track, primaryConnectionExtrapDistance_min, primaryConnectionExtrapDistance_max)
    normaliseXAxis(primaryClosestParentL_track, primaryClosestParentL_min, primaryClosestParentL_max) 
    normaliseXAxis(primaryClosestParentT_track, primaryClosestParentT_min, primaryClosestParentT_max)    
    
    normaliseXAxis(primaryNSpacepoints_shower, primaryNSpacepoints_min, primaryNSpacepoints_max)    
    normaliseXAxis(primaryNuVertexSeparation_shower, primaryNuVertexSeparation_min, primaryNuVertexSeparation_max)    
    normaliseXAxis(primaryStartRegionNHits_shower, primaryStartRegionNHits_min, primaryStartRegionNHits_max)    
    normaliseXAxis(primaryStartRegionNParticles_shower, primaryStartRegionNParticles_min, primaryStartRegionNParticles_max)    
    normaliseXAxis(primaryDCA_shower, primaryDCA_min, primaryDCA_max)
    normaliseXAxis(primaryConnectionExtrapDistance_shower, primaryConnectionExtrapDistance_min, primaryConnectionExtrapDistance_max)
    normaliseXAxis(primaryClosestParentL_shower, primaryClosestParentL_min, primaryClosestParentL_max) 
    normaliseXAxis(primaryClosestParentT_shower, primaryClosestParentT_min, primaryClosestParentT_max)      
    
    ###################################
    # How many links do we have?
    ###################################        
    nLinks_track = primaryPFPIndex_track.shape[0] 
    nLinks_shower = primaryPFPIndex_shower.shape[0]
    
    ###################################
    # Concatenate
    ###################################
    coc0_track = np.concatenate((np.concatenate((primaryNuVertexSeparation_track[0, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNHits_track[0, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNParticles_track[0, :].reshape(nLinks_track, 1), \
                                           primaryDCA_track[0, :].reshape(nLinks_track, 1), \
                                           primaryConnectionExtrapDistance_track[0, :].reshape(nLinks_track, 1), \
                                           primaryIsPOIClosestToNu_track[0, :].reshape(nLinks_track, 1), \
                                           primaryClosestParentL_track[0, :].reshape(nLinks_track, 1), \
                                           primaryClosestParentT_track[0, :].reshape(nLinks_track, 1)), axis=1), \
                           np.concatenate((primaryNuVertexSeparation_track[1, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNHits_track[1, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNParticles_track[1, :].reshape(nLinks_track, 1), \
                                           primaryDCA_track[1, :].reshape(nLinks_track, 1), \
                                           primaryConnectionExtrapDistance_track[1, :].reshape(nLinks_track, 1), \
                                           primaryIsPOIClosestToNu_track[1, :].reshape(nLinks_track, 1), \
                                           primaryClosestParentL_track[1, :].reshape(nLinks_track, 1), \
                                           primaryClosestParentT_track[1, :].reshape(nLinks_track, 1)), axis=1)), axis=1)   
    
    # concatenate variable_single and orientations
    variables_track = np.concatenate((primaryNSpacepoints_track.reshape(nLinks_track, 1), \
                                coc0_track), axis=1)
    
    variables_shower = np.concatenate((primaryNSpacepoints_shower.reshape(nLinks_shower, 1), \
                                       primaryNuVertexSeparation_shower.reshape(nLinks_shower, 1), \
                                       primaryStartRegionNHits_shower.reshape(nLinks_shower, 1), \
                                       primaryStartRegionNParticles_shower.reshape(nLinks_shower, 1), \
                                       primaryDCA_shower.reshape(nLinks_shower, 1), \
                                       primaryConnectionExtrapDistance_shower.reshape(nLinks_shower, 1), \
                                       primaryIsPOIClosestToNu_shower.reshape(nLinks_shower, 1), \
                                       primaryClosestParentL_shower.reshape(nLinks_shower, 1), \
                                       primaryClosestParentT_shower.reshape(nLinks_shower, 1)), axis=1)
    
    return primaryPFPIndex_track, variables_track, y_track, isTruePrimaryLink_track, trainingDCA_track, primaryPFPIndex_shower, variables_shower, isTruePrimaryLink_shower, trainingDCA_shower

    
############################################################################################################################################
############################################################################################################################################         

    