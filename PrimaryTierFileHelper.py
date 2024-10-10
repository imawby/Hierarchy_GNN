import numpy as np
import uproot
import math

from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

###################################
###################################
nuVertexEnergyAsymmetry_min = -1.0
nuVertexEnergyAsymmetry_max = 1.0
nuVertexEnergyWeightedMeanRadialDistance_min = -15.0
nuVertexEnergyWeightedMeanRadialDistance_max = 20.0
nViewsWithAmbiguousHits_min = -1
nViewsWithAmbiguousHits_max = 3
ambiguousUnaccountedEnergy_min = -15.0
ambiguousUnaccountedEnergy_max = 15.0
displacement_min = -10.0
displacement_max = 100.0
dca_min = -10.0
dca_max = 50.0

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

###################################
###################################

def readTree(fileNames) :
    
    ###################################
    # To pull out of tree
    ###################################
    # Link variables
    nuVertexEnergyAsymmetry = []
    nuVertexEnergyWeightedMeanRadialDistance = []
    nViewsWithAmbiguousHits = []
    ambiguousUnaccountedEnergy = []
    displacement = []
    dca = []
    # Truth
    isHierarchyTrainingNode = []
    trueVisibleGeneration = []
    
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
            
            # Edge information
            nuVertexEnergyAsymmetry.extend(branches["RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyAsymmetry"][iEvent])
            nuVertexEnergyWeightedMeanRadialDistance.extend(branches["RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyWeightedMeanRadialDistance"][iEvent])
            nViewsWithAmbiguousHits.extend(branches["RecoShowerPandrizzleNViewsWithAmbiguousHits"][iEvent])
            ambiguousUnaccountedEnergy.extend(branches["RecoShowerPandrizzleAmbiguousHitMaxUnaccountedEnergy"][iEvent])
            displacement.extend(branches["RecoShowerPandrizzleDisplacement"][iEvent])
            dca.extend(branches["RecoShowerPandrizzleDCA"][iEvent])
            # Truth
            isHierarchyTrainingNode.extend(branches["IsHierarchyTrainingNode"][iEvent])
            trueVisibleGeneration.extend(branches["RecoPFPTrueVisibleGeneration"][iEvent])     
                    
    ###################################
    # Now turn things into numpy arrays
    ###################################
    # Link variables
    nuVertexEnergyAsymmetry = np.array(nuVertexEnergyAsymmetry)
    nuVertexEnergyWeightedMeanRadialDistance = np.array(nuVertexEnergyWeightedMeanRadialDistance)
    nViewsWithAmbiguousHits = np.array(nViewsWithAmbiguousHits)
    ambiguousUnaccountedEnergy = np.array(ambiguousUnaccountedEnergy)
    displacement = np.array(displacement)
    dca = np.array(dca)
    # Truth
    isHierarchyTrainingNode = np.array(isHierarchyTrainingNode)
    trueVisibleGeneration = np.array(trueVisibleGeneration)
    
    ###################################
    # Only consider training links!
    ###################################
    isHierarchyTrainingNodeMask = (isHierarchyTrainingNode == True)
    # Link variables
    nuVertexEnergyAsymmetry = nuVertexEnergyAsymmetry[isHierarchyTrainingNodeMask]
    nuVertexEnergyWeightedMeanRadialDistance = nuVertexEnergyWeightedMeanRadialDistance[isHierarchyTrainingNodeMask]
    nViewsWithAmbiguousHits = nViewsWithAmbiguousHits[isHierarchyTrainingNodeMask]
    ambiguousUnaccountedEnergy = ambiguousUnaccountedEnergy[isHierarchyTrainingNodeMask]
    displacement = displacement[isHierarchyTrainingNodeMask]
    dca = dca[isHierarchyTrainingNodeMask]
    # Truth
    trueVisibleGeneration = trueVisibleGeneration[isHierarchyTrainingNodeMask]
        
    nLinks = trueVisibleGeneration.shape[0]   
    print('We have ', str(nLinks), ' to train on!')
    
    ###################################
    # Normalise variables
    ###################################
    normaliseXAxis(nuVertexEnergyAsymmetry, nuVertexEnergyAsymmetry_min, nuVertexEnergyAsymmetry_max)
    normaliseXAxis(nuVertexEnergyWeightedMeanRadialDistance, nuVertexEnergyWeightedMeanRadialDistance_min, nuVertexEnergyWeightedMeanRadialDistance_max)
    normaliseXAxis(nViewsWithAmbiguousHits, nViewsWithAmbiguousHits_min, nViewsWithAmbiguousHits_max)
    normaliseXAxis(ambiguousUnaccountedEnergy, ambiguousUnaccountedEnergy_min, ambiguousUnaccountedEnergy_max)
    normaliseXAxis(displacement, displacement_min, displacement_max)
    normaliseXAxis(dca, dca_min, dca_max)

    ###################################
    # Reshape
    ###################################
    # Node variables
    nuVertexEnergyAsymmetry = nuVertexEnergyAsymmetry.reshape((nLinks, 1))
    nuVertexEnergyWeightedMeanRadialDistance = nuVertexEnergyWeightedMeanRadialDistance.reshape((nLinks, 1))
    nViewsWithAmbiguousHits = nViewsWithAmbiguousHits.reshape((nLinks, 1))
    ambiguousUnaccountedEnergy = ambiguousUnaccountedEnergy.reshape((nLinks, 1))
    displacement = displacement.reshape((nLinks, 1))
    dca = dca.reshape((nLinks, 1))
    # Truth
    trueVisibleGeneration = trueVisibleGeneration.reshape((nLinks, 1))
    
    ###################################
    # Concatenate
    ###################################          
    variables = np.concatenate((nuVertexEnergyAsymmetry, \
                                nuVertexEnergyWeightedMeanRadialDistance, \
                                nViewsWithAmbiguousHits, \
                                ambiguousUnaccountedEnergy, \
                                displacement, \
                                dca), axis=1)

    ###################################
    # Convert truth vector
    ################################### 
    y = trueVisibleGeneration
    
    return nLinks, variables, y


############################################################################################################################################
############################################################################################################################################

def normaliseXAxis(variable, minLimit, maxLimit) :

    interval = math.fabs(minLimit) + math.fabs(maxLimit)
    variable[variable < minLimit] = minLimit
    variable[variable > maxLimit] = maxLimit
    variable /= interval
    
    
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
            # True
            isTruePrimaryLink_file = np.array(branches['TruePrimaryLink'][iEvent][primaryTrack_mask])
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
                    # set the common vars
                    primaryNSpacepoints.append(primaryNSpacepoints_file[iLink])   
                    isTruePrimaryLink.append(isTruePrimaryLink_file[iLink])
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
    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
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
        
    ###################################
    # Concatenate
    ###################################
    coc0 = np.concatenate((np.concatenate((primaryNuVertexSeparation[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[0, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[0, :].reshape(nLinks, 1), \
                                           primaryDCA[0, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[0, :].reshape(nLinks, 1)), axis=1), \
                           np.concatenate((primaryNuVertexSeparation[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNHits[1, :].reshape(nLinks, 1), \
                                           primaryStartRegionNParticles[1, :].reshape(nLinks, 1), \
                                           primaryDCA[1, :].reshape(nLinks, 1), \
                                           primaryConnectionExtrapDistance[1, :].reshape(nLinks, 1)), axis=1)), axis=1)      
    
    
    
    # concatenate variable_single and orientations
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                                coc0), axis=1)

    
    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect

    
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
            primaryNSpacepoints.extend(branches['PrimaryNSpacepoints'][iEvent][primaryShower_mask])
            primaryNuVertexSeparation.extend(branches['PrimaryNuVertexSeparation'][iEvent][primaryShower_mask])
            primaryStartRegionNHits.extend(branches['PrimaryStartRegionNHits'][iEvent][primaryShower_mask])            
            primaryStartRegionNParticles.extend(branches['PrimaryStartRegionNParticles'][iEvent][primaryShower_mask])            
            primaryDCA.extend(branches['PrimaryDCA'][iEvent][primaryShower_mask])            
            primaryConnectionExtrapDistance.extend(branches['PrimaryConnectionExtrapDistance'][iEvent][primaryShower_mask])
            # True
            isTruePrimaryLink_file = branches['TruePrimaryLink'][iEvent][primaryShower_mask]
            isTruePrimaryLink.extend(isTruePrimaryLink_file)
            isLinkOrientationCorrect_file = branches['IsPrimaryLinkOrientationCorrect'][iEvent][primaryShower_mask]
            isLinkOrientationCorrect.extend(isLinkOrientationCorrect_file)
           
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
    # Truth
    isTruePrimaryLink = np.array(isTruePrimaryLink)
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
        
        
    ###################################
    # Concatenate
    ###################################
    variables = np.concatenate((primaryNSpacepoints.reshape(nLinks, 1), \
                           primaryNuVertexSeparation.reshape(nLinks, 1), \
                           primaryStartRegionNHits.reshape(nLinks, 1), \
                           primaryStartRegionNParticles.reshape(nLinks, 1), \
                           primaryDCA.reshape(nLinks, 1), \
                           primaryConnectionExtrapDistance.reshape(nLinks, 1)), axis=1)    

    
    return nLinks, variables, y, isTruePrimaryLink, isLinkOrientationCorrect
        
        
############################################################################################################################################
############################################################################################################################################        

def readEvent(eventDict) :
        
    ###################################
    # Need to normalise!
    ###################################    
    normaliseXAxis(eventDict['primaryNSpacepoints'], primaryNSpacepoints_min, primaryNSpacepoints_max)    
    normaliseXAxis(eventDict['primaryNuVertexSeparation'], primaryNuVertexSeparation_min, primaryNuVertexSeparation_max)    
    normaliseXAxis(eventDict['primaryStartRegionNHits'], primaryStartRegionNHits_min, primaryStartRegionNHits_max)    
    normaliseXAxis(eventDict['primaryStartRegionNParticles'], primaryStartRegionNParticles_min, primaryStartRegionNParticles_max)    
    normaliseXAxis(eventDict['primaryDCA'], primaryDCA_min, primaryDCA_max)
    normaliseXAxis(eventDict['primaryConnectionExtrapDistance'], primaryConnectionExtrapDistance_min, primaryConnectionExtrapDistance_max)               
        
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
    # ID
    primaryPFPIndex_track = []
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
    # ID
    primaryPFPIndex_shower = []
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

        linksMadeCounter = linksMadeCounter + 1

        if (isTrack and (linksMadeCounter == 2)) :
            y_track.append(this_y_track)                       
            linksMadeCounter = 0
            this_y_track = [0, 0]
            
        if (not isTrack) :
            linksMadeCounter = 0
            
    ###################################
    # Now turn things into numpy arrays
    ###################################
    primaryTrackScore_track = np.array(primaryTrackScore_track, dtype='float64')
    primaryNSpacepoints_track = np.array(primaryNSpacepoints_track, dtype='float64')
    primaryNuVertexSeparation_track = np.array(primaryNuVertexSeparation_track, dtype='float64')
    primaryStartRegionNHits_track = np.array(primaryStartRegionNHits_track, dtype='float64')
    primaryStartRegionNParticles_track = np.array(primaryStartRegionNParticles_track, dtype='float64')
    primaryDCA_track = np.array(primaryDCA_track, dtype='float64')
    primaryConnectionExtrapDistance_track = np.array(primaryConnectionExtrapDistance_track, dtype='float64')
    primaryPFPIndex_track = np.array(primaryPFPIndex_track, dtype='int')
    isTruePrimaryLink_track = np.array(isTruePrimaryLink_track, dtype='int')
    y_track = np.array(y_track, dtype='int')
            
    primaryTrackScore_shower = np.array(primaryTrackScore_shower, dtype='float64')
    primaryNSpacepoints_shower = np.array(primaryNSpacepoints_shower, dtype='float64')
    primaryNuVertexSeparation_shower = np.array(primaryNuVertexSeparation_shower, dtype='float64')
    primaryStartRegionNHits_shower = np.array(primaryStartRegionNHits_shower, dtype='float64')
    primaryStartRegionNParticles_shower = np.array(primaryStartRegionNParticles_shower, dtype='float64')
    primaryDCA_shower = np.array(primaryDCA_shower, dtype='float64')
    primaryConnectionExtrapDistance_shower = np.array(primaryConnectionExtrapDistance_shower, dtype='float64')
    primaryPFPIndex_shower = np.array(primaryPFPIndex_shower, dtype='int')
    isTruePrimaryLink_shower = np.array(isTruePrimaryLink_shower, dtype='int')

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
                                           primaryConnectionExtrapDistance_track[0, :].reshape(nLinks_track, 1)), axis=1), \
                           np.concatenate((primaryNuVertexSeparation_track[1, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNHits_track[1, :].reshape(nLinks_track, 1), \
                                           primaryStartRegionNParticles_track[1, :].reshape(nLinks_track, 1), \
                                           primaryDCA_track[1, :].reshape(nLinks_track, 1), \
                                           primaryConnectionExtrapDistance_track[1, :].reshape(nLinks_track, 1)), axis=1)), axis=1)   
    
    # concatenate variable_single and orientations
    variables_track = np.concatenate((primaryNSpacepoints_track.reshape(nLinks_track, 1), \
                                coc0_track), axis=1)
    
    variables_shower = np.concatenate((primaryNSpacepoints_shower.reshape(nLinks_shower, 1), \
                                       primaryNuVertexSeparation_shower.reshape(nLinks_shower, 1), \
                                       primaryStartRegionNHits_shower.reshape(nLinks_shower, 1), \
                                       primaryStartRegionNParticles_shower.reshape(nLinks_shower, 1), \
                                       primaryDCA_shower.reshape(nLinks_shower, 1), \
                                       primaryConnectionExtrapDistance_shower.reshape(nLinks_shower, 1)), axis=1)
    
    return primaryPFPIndex_track, variables_track, y_track, isTruePrimaryLink_track, primaryPFPIndex_shower, variables_shower, isTruePrimaryLink_shower

    
############################################################################################################################################
############################################################################################################################################         

    