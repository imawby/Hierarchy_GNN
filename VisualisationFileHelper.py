import numpy as np
import uproot
import awkward as ak

def readTree(fileName, targetRun, targetSubrun, targetEvent) :
    
    treeFile = uproot.open(fileName)
    tree = treeFile['ccnuselection/ccnusel']
    branches = tree.arrays()
    
    #############################
    # Get event-level stuff
    #############################
    run = np.array(branches['Run'])
    subrun = np.array(branches['SubRun'])
    event = np.array(branches['Event'])
    nEntries = run.shape[0]
    
    isNC = np.array(branches['NC'])
    nuPDG = np.array(branches['NuPdg'])
    recoNuX = np.array(branches['RecoNuVtxX'])
    recoNuY = np.array(branches['RecoNuVtxY'])
    recoNuZ = np.array(branches['RecoNuVtxZ'])  
    
    #############################
    # Get pfp-level stuff - these cannot be numpy arrays...
    #############################
    pfpSelf = branches['RecoPFPSelf']  
    pfpRecoGeneration = branches['RecoPFPRecoGeneration']
    pfpRecoParentSelf = branches['RecoPFPRecoParentSelf']
    
    trackRecoStartX = branches['RecoTrackRecoStartX']
    trackRecoStartY = branches['RecoTrackRecoStartY']
    trackRecoStartZ = branches['RecoTrackRecoStartZ']
    pfpSpacepointX = branches['RecoPFPSpacepointX']
    pfpSpacepointY = branches['RecoPFPSpacepointY']
    pfpSpacepointZ = branches['RecoPFPSpacepointZ']

    pfpTrackShowerScore = branches['RecoPFPTrackShowerScore']
    pfpNHits = branches['RecoPFPRecoNHits']
    pfpCharge = branches['RecoPFPRecoCharge']
    pfpRecoVertexX = branches['RecoPFPRecoVertexX']
    pfpRecoVertexY = branches['RecoPFPRecoVertexY']
    pfpRecoVertexZ = branches['RecoPFPRecoVertexZ']
    trackRecoEndX = branches['RecoTrackRecoEndX']
    trackRecoEndY = branches['RecoTrackRecoEndY']
    trackRecoEndZ = branches['RecoTrackRecoEndZ']
    showerDirX = branches['RecoShowerRecoDirX']  # not the best direction estimate, placeholder
    showerDirY = branches['RecoShowerRecoDirY']
    showerDirZ = branches['RecoShowerRecoDirZ']
    pfpIvysaurusMuon = branches['RecoPFPIvysaurusMuon']
    pfpIvysaurusProton = branches['RecoPFPIvysaurusProton']
    pfpIvysaurusPion = branches['RecoPFPIvysaurusPion']
    pfpIvysaurusPhoton = branches['RecoPFPIvysaurusPhoton']
    pfpIvysaurusElectron = branches['RecoPFPIvysaurusElectron']
    pfpTrackLength = branches['RecoTrackLength']
    pfpDisplacement = branches['RecoShowerPandrizzleDisplacement']
    pfpDCA = branches['RecoShowerPandrizzleDCA']
    pfpNuVertexEnergyAsymmetry = branches['RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyAsymmetry']
    pfpNuVertexEnergyWeightedMeanRadialDistance = branches['RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyWeightedMeanRadialDistance']
    
    pfpTrueMomX = branches['RecoPFPTrueMomX']
    pfpTrueMomY = branches['RecoPFPTrueMomY']
    pfpTrueMomZ = branches['RecoPFPTrueMomZ']
    pfpTrueTrackID = branches['RecoPFPTrueTrackID']
    pfpCompleteness = branches['RecoPFPRecoCompleteness']
    pfpPurity = branches['RecoPFPRecoHitPurity']
    pfpTruePDG = branches['RecoPFPTruePDG']
    pfpTrueGeneration = branches['RecoPFPTrueGeneration']
    pfpTrueParentTrackID = branches['RecoPFPTrueParentTrackID']
    pfpTrueParentPDG = branches['RecoPFPTrueParentPDG']
    pfpTrueVisibleGeneration = branches['RecoPFPTrueVisibleGeneration']
    pfpTrueVisibleParentTrackID = branches['RecoPFPTrueVisibleParentTrackID']
    pfpTrueVisibleParentPDG = branches['RecoPFPTrueVisibleParentPDG']

    # Get index of correct run, subrun, event
    indicies = np.array(range(nEntries))
    runMask = (run == targetRun)
    
    if (np.count_nonzero(runMask) == 0) :
        print('Run not found :(')
    
    subrunMask = (subrun == targetSubrun)
    
    if (np.count_nonzero(subrunMask) == 0) :
        print('Subrun not found :(')
    
    eventMask = (event == targetEvent)
    
    if (np.count_nonzero(eventMask) == 0) :
        print('event not found :(')
    
    mask = np.logical_and(runMask, np.logical_and(subrunMask, eventMask))
    print('This should be one: ', np.count_nonzero(mask))
    
    targetIndex = indicies[mask][0]
    
    # Get the info for the event we care about    
    isNC = isNC[targetIndex]
    nuPDG = nuPDG[targetIndex]
    recoNuX = recoNuX[targetIndex]
    recoNuY = recoNuY[targetIndex]
    recoNuZ = recoNuZ[targetIndex]
    
    pfpSelf = np.array(pfpSelf[targetIndex])
    pfpRecoGeneration = np.array(pfpRecoGeneration[targetIndex])
    pfpRecoParentSelf = np.array(pfpRecoParentSelf[targetIndex])
    
    trackRecoStartX = np.array(trackRecoStartX[targetIndex])
    trackRecoStartY = np.array(trackRecoStartY[targetIndex])
    trackRecoStartZ = np.array(trackRecoStartZ[targetIndex])
    pfpSpacepointX = pfpSpacepointX[targetIndex]
    pfpSpacepointY = pfpSpacepointY[targetIndex]
    pfpSpacepointZ = pfpSpacepointZ[targetIndex]
    pfpTrackShowerScore = np.array(pfpTrackShowerScore[targetIndex])
    pfpNHits = np.array(pfpNHits[targetIndex])
    pfpCharge = np.array(pfpCharge[targetIndex])
    pfpRecoVertexX = np.array(pfpRecoVertexX[targetIndex])
    pfpRecoVertexY = np.array(pfpRecoVertexY[targetIndex])
    pfpRecoVertexZ = np.array(pfpRecoVertexZ[targetIndex])
    trackRecoEndX = np.array(trackRecoEndX[targetIndex])
    trackRecoEndY = np.array(trackRecoEndY[targetIndex])
    trackRecoEndZ = np.array(trackRecoEndZ[targetIndex])
    showerDirX = np.array(showerDirX[targetIndex])
    showerDirY = np.array(showerDirY[targetIndex])
    showerDirZ = np.array(showerDirZ[targetIndex])
    pfpIvysaurusMuon = np.array(pfpIvysaurusMuon[targetIndex])
    pfpIvysaurusProton = np.array(pfpIvysaurusProton[targetIndex])
    pfpIvysaurusPion = np.array(pfpIvysaurusPion[targetIndex])
    pfpIvysaurusPhoton = np.array(pfpIvysaurusPhoton[targetIndex])
    pfpIvysaurusElectron = np.array(pfpIvysaurusElectron[targetIndex])
    pfpTrackLength = np.array(pfpTrackLength[targetIndex])
    pfpDisplacement = np.array(pfpDisplacement[targetIndex])
    pfpDCA = np.array(pfpDCA[targetIndex])
    pfpIsNeutrinoPDG = np.zeros(pfpTrackShowerScore.shape)
    pfpNuVertexEnergyAsymmetry = np.array(pfpNuVertexEnergyAsymmetry[targetIndex])
    pfpNuVertexEnergyWeightedMeanRadialDistance = np.array(pfpNuVertexEnergyWeightedMeanRadialDistance[targetIndex])
    
    pfpTrueMomX = np.array(pfpTrueMomX[targetIndex])
    pfpTrueMomY = np.array(pfpTrueMomY[targetIndex])
    pfpTrueMomZ = np.array(pfpTrueMomZ[targetIndex])
    pfpTrueTrackID = np.array(pfpTrueTrackID[targetIndex])
    pfpCompleteness = np.array(pfpCompleteness[targetIndex])
    pfpPurity = np.array(pfpPurity[targetIndex])
    pfpTruePDG = np.array(pfpTruePDG[targetIndex])
    pfpTrueGeneration = np.array(pfpTrueGeneration[targetIndex])
    pfpTrueParentTrackID = np.array(pfpTrueParentTrackID[targetIndex])
    pfpTrueParentPDG = np.array(pfpTrueParentPDG[targetIndex])
    pfpTrueVisibleGeneration = np.array(pfpTrueVisibleGeneration[targetIndex])
    pfpTrueVisibleParentTrackID = np.array(pfpTrueVisibleParentTrackID[targetIndex])
    pfpTrueVisibleParentPDG = np.array(pfpTrueVisibleParentPDG[targetIndex])

    return isNC, nuPDG, recoNuX, recoNuY, recoNuZ, pfpSelf, pfpRecoGeneration, pfpRecoParentSelf, trackRecoStartX, trackRecoStartY, trackRecoStartZ, pfpSpacepointX, pfpSpacepointY, pfpSpacepointZ, pfpTrackShowerScore, pfpNHits, pfpCharge, pfpRecoVertexX, pfpRecoVertexY, pfpRecoVertexZ, trackRecoEndX, trackRecoEndY, trackRecoEndZ, showerDirX, showerDirY, showerDirZ, pfpIvysaurusMuon, pfpIvysaurusProton, pfpIvysaurusPion, pfpIvysaurusPhoton, pfpIvysaurusElectron, pfpTrackLength, pfpDisplacement, pfpDCA, pfpIsNeutrinoPDG, pfpNuVertexEnergyAsymmetry, pfpNuVertexEnergyWeightedMeanRadialDistance, pfpTrueMomX, pfpTrueMomY, pfpTrueMomZ, pfpTrueTrackID, pfpCompleteness, pfpPurity, pfpTruePDG, pfpTrueGeneration, pfpTrueParentTrackID, pfpTrueParentPDG, pfpTrueVisibleGeneration, pfpTrueVisibleParentTrackID, pfpTrueVisibleParentPDG