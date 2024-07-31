import numpy as np
import uproot

from tensorflow.keras.utils import to_categorical


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
    # Normalise track vars
    ###################################
    
    ###################################
    # Concatenate
    ###################################          
    variables = np.concatenate((parentTrackScore, parentNuVertexSeparation, childNuVertexSeparation, parentEndRegionNHits, parentEndRegionNParticles, parentEndRegionRToWall, vertexSeparation, separation3D, chargeRatio, pidLinkType, openingAngle, trackShowerLinkType), axis=1)

    ###################################
    # Convert truth vector
    ################################### 
    y = trueParentChildLink #to_categorical(trueParentChildLink, 2) # true/false
    
    return nLinks, variables, y