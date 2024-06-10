import math
import numpy as np
import sys
import torch
from torch_geometric.data import Data

import Normalisation

#############################
# Graph Builder Function
#############################
def GraphBuilder(eventDict, modeDict) :
        
    # Our lists that we want to fill
    event_node_features = []
    
    source_index_pos = []
    source_index_neg = []
    
    target_index_pos = []
    target_index_neg = []    
    
    source_index_FC = []
    target_index_FC = []
    
    pfpIndex = []
    
    print(type(eventDict["vertexX"]))
    
    # Because i'm an idiot and set the vertex default value bad
    failedVertexMask = eventDict["vertexX"] < -990.0
    eventDict["vertexX"][failedVertexMask] = -9999.0
    eventDict["vertexY"][failedVertexMask] = -9999.0
    eventDict["vertexZ"][failedVertexMask] = -9999.0
    failedEndpointMask = eventDict["trackEndX"] < -990.0
    eventDict["trackEndX"][failedEndpointMask] = -9999.0
    eventDict["trackEndY"][failedEndpointMask] = -9999.0
    eventDict["trackEndZ"][failedEndpointMask] = -9999.0
    
    print(type(eventDict["vertexX"]))
    
    # I need to add in the neutrino (i need to put this in my analyser?)
    if modeDict["ADD_NEUTRINO"] :
        # For node
        eventDict["trackShowerScore"]                         = np.array(np.append(eventDict["trackShowerScore"], -999))
        eventDict["nHits"]                                    = np.array(np.append(eventDict["nHits"], -999))
        eventDict["charge"]                                   = np.array(np.append(eventDict["charge"], -999.0))
        eventDict["vertexX"]                                  = np.array(np.append(eventDict["vertexX"], eventDict["recoNuVertexX"]))
        eventDict["vertexY"]                                  = np.array(np.append(eventDict["vertexY"], eventDict["recoNuVertexY"]))
        eventDict["vertexZ"]                                  = np.array(np.append(eventDict["vertexZ"], eventDict["recoNuVertexZ"]))
        eventDict["trackEndX"]                                = np.array(np.append(eventDict["trackEndX"], eventDict["recoNuVertexX"]))
        eventDict["trackEndY"]                                = np.array(np.append(eventDict["trackEndY"], eventDict["recoNuVertexY"]))
        eventDict["trackEndZ"]                                = np.array(np.append(eventDict["trackEndZ"], eventDict["recoNuVertexZ"]))
        eventDict["showerDirX"]                               = np.array(np.append(eventDict["showerDirX"], -999.0))
        eventDict["showerDirY"]                               = np.array(np.append(eventDict["showerDirY"], -999.0))
        eventDict["showerDirZ"]                               = np.array(np.append(eventDict["showerDirZ"], -999.0))
        eventDict["ivysaurusMuon"]                            = np.array(np.append(eventDict["ivysaurusMuon"], -999.0))
        eventDict["ivysaurusProton"]                          = np.array(np.append(eventDict["ivysaurusProton"], -999.0))
        eventDict["ivysaurusPion"]                            = np.array(np.append(eventDict["ivysaurusPion"], -999.0))
        eventDict["ivysaurusElectron"]                        = np.array(np.append(eventDict["ivysaurusElectron"], -999.0))
        eventDict["ivysaurusPhoton"]                          = np.array(np.append(eventDict["ivysaurusPhoton"], -999.0))
        eventDict["trackLength"]                              = np.array(np.append(eventDict["trackLength"], -999.0))
        eventDict["displacement"]                             = np.array(np.append(eventDict["displacement"], -999.0))
        eventDict["dca"]                                      = np.array(np.append(eventDict["dca"], -999.0))
        eventDict["isNeutrinoPDG"]                            = np.array(np.append(eventDict["isNeutrinoPDG"], int(1)))
        eventDict["nuVertexEnergyAsymmetry"]                  = np.array(np.append(eventDict["nuVertexEnergyAsymmetry"], -999.0))
        eventDict["nuVertexEnergyWeightedMeanRadialDistance"] = np.array(np.append(eventDict["nuVertexEnergyWeightedMeanRadialDistance"], -999.0))
        # For training
        eventDict["trueTrackID"]                              = np.array(np.append(eventDict["trueTrackID"], 0))
        eventDict["truePDG"]                                  = np.array(np.append(eventDict["truePDG"], -1))
        eventDict["trueVisibleParentTrackID"]                 = np.array(np.append(eventDict["trueVisibleParentTrackID"], -999))
        
        # True information.. (for cheating)
        if modeDict["CHEAT_DIRECTION"] :
            eventDict["trueMomX"]                          = np.array(np.append(eventDict["trueMomX"], -999.0))
            eventDict["trueMomY"]                          = np.array(np.append(eventDict["trueMomY"], -999.0))
            eventDict["trueMomZ"]                          = np.array(np.append(eventDict["trueMomZ"], -999.0))
     
    print(type(eventDict["vertexX"]))
    
    # We need not normalised versions to determine particle-particle links 
    vertex_notNorm = np.concatenate((eventDict["vertexX"].reshape(-1,1), eventDict["vertexY"].reshape(-1,1), eventDict["vertexZ"].reshape(-1,1)), axis=1)
    trackEnd_notNorm = np.concatenate((eventDict["trackEndX"].reshape(-1,1), eventDict["trackEndY"].reshape(-1,1), eventDict["trackEndZ"].reshape(-1,1)), axis=1)
    showerDir_notNorm = np.concatenate((eventDict["showerDirX"].reshape(-1,1), eventDict["showerDirY"].reshape(-1,1), eventDict["showerDirZ"].reshape(-1,1)), axis=1)
    
    if modeDict["CHEAT_DIRECTION"] :
        # Replace shower direction with truth
        noTrueDirMask = np.logical_or(eventDict["trueMomX"] < -990, np.logical_or(eventDict["trueMomY"] < -990, eventDict["trueMomZ"] < -990))
        pfpTrueDir_np = np.concatenate((eventDict["trueMomX"].reshape(-1,1), eventDict["trueMomY"].reshape(-1,1), 
                                        eventDict["trueMomZ"].reshape(-1,1)), axis=1)
        mag = np.linalg.norm(pfpTrueDir_np, axis=1)
 
        eventDict["showerDirX"] = eventDict["trueMomX"] / mag
        eventDict["showerDirY"] = eventDict["trueMomY"] / mag
        eventDict["showerDirZ"] = eventDict["trueMomZ"] / mag

        # Remember to keep the ones that failed
        eventDict["showerDirX"][noTrueDirMask] = -999.0
        eventDict["showerDirY"][noTrueDirMask] = -999.0
        eventDict["showerDirZ"][noTrueDirMask] = -999.0

        # Recalculate the DCA (because it relies on shower direction)
        showerDir_notNorm = np.concatenate((eventDict["showerDirX"].reshape(-1,1), eventDict["showerDirY"].reshape(-1,1), 
                                            eventDict["showerDirZ"].reshape(-1,1)), axis=1)

        nuVertexX_notNorm = (np.ones(vertex_notNorm.shape[0]) * eventDict["recoNuVertexX"]).reshape(-1,1)
        nuVertexY_notNorm = (np.ones(vertex_notNorm.shape[0]) * eventDict["recoNuVertexY"]).reshape(-1,1)
        nuVertexZ_notNorm = (np.ones(vertex_notNorm.shape[0]) * eventDict["recoNuVertexZ"]).reshape(-1,1)
        nuVertex_notNorm = np.concatenate((nuVertexX_notNorm.reshape(-1,1), nuVertexY_notNorm.reshape(-1,1), nuVertexZ_notNorm.reshape(-1,1)), axis=1)
        displacementVec = vertex_notNorm - nuVertex_notNorm
        
        noDCAMask = eventDict["dca"] < -990
        eventDict["dca"] = np.linalg.norm(np.cross(displacementVec, showerDir_notNorm), axis=1)
        eventDict["dca"][noDCAMask] = -999.0
        
    if modeDict["CHEAT_PID"] :
        eventDict["ivysaurusMuon"][np.abs(eventDict["truePDG"]) == 13] = 1
        eventDict["ivysaurusMuon"][np.abs(eventDict["truePDG"]) != 13] = -1 # because the neutrino has 0 taken
        eventDict["ivysaurusProton"][np.abs(eventDict["truePDG"]) == 2212] = 1
        eventDict["ivysaurusProton"][np.abs(eventDict["truePDG"]) != 2212] = -1
        eventDict["ivysaurusPion"][np.abs(eventDict["truePDG"]) == 211] = 1
        eventDict["ivysaurusPion"][np.abs(eventDict["truePDG"]) != 211] = -1
        eventDict["ivysaurusElectron"][np.abs(eventDict["truePDG"]) == 11] = 1
        eventDict["ivysaurusElectron"][np.abs(eventDict["truePDG"]) != 11] = -1
        eventDict["ivysaurusPhoton"][np.abs(eventDict["truePDG"]) == 22] = 1
        eventDict["ivysaurusPhoton"][np.abs(eventDict["truePDG"]) != 22] = -1
        
    # Normalise
    if modeDict["DO_NORMALISATION"] :
        Normalisation.normaliseTrackShowerScore(eventDict["trackShowerScore"])
        Normalisation.normaliseNHits(eventDict["nHits"])
        Normalisation.normaliseCharge(eventDict["charge"])
        Normalisation.normalisePositionX(eventDict["vertexX"])
        Normalisation.normalisePositionY(eventDict["vertexY"])
        Normalisation.normalisePositionZ(eventDict["vertexZ"])
        Normalisation.normalisePositionX(eventDict["trackEndX"])
        Normalisation.normalisePositionY(eventDict["trackEndY"])
        Normalisation.normalisePositionZ(eventDict["trackEndZ"])
        Normalisation.normaliseShowerDir(eventDict["showerDirX"])
        Normalisation.normaliseShowerDir(eventDict["showerDirY"])
        Normalisation.normaliseShowerDir(eventDict["showerDirZ"])
        Normalisation.normaliseIvysaurusScore(eventDict["ivysaurusMuon"])
        Normalisation.normaliseIvysaurusScore(eventDict["ivysaurusProton"])
        Normalisation.normaliseIvysaurusScore(eventDict["ivysaurusPion"])
        Normalisation.normaliseIvysaurusScore(eventDict["ivysaurusElectron"])
        Normalisation.normaliseIvysaurusScore(eventDict["ivysaurusPhoton"])
        Normalisation.normaliseTrackLength(eventDict["trackLength"])
        Normalisation.normaliseDisplacement(eventDict["displacement"])
        Normalisation.normaliseDCA(eventDict["dca"])
        Normalisation.normaliseNuVertexEnergyAsymmetry(eventDict["nuVertexEnergyAsymmetry"])
        Normalisation.normaliseNuVertexEnergyWeightedMeanRadialDistance(eventDict["nuVertexEnergyWeightedMeanRadialDistance"])
        
    # Put these here so we pick up the added neutrino
    nGraphNodes = 0
    nPosEdge = 0
    nNegEdge = 0
    
    for iSourceParticle in range(eventDict["nParticles"]) :
                
        # Node features
        trackShowerScore = eventDict["trackShowerScore"][iSourceParticle]
        nHits = eventDict["nHits"][iSourceParticle]
        charge = eventDict["charge"][iSourceParticle]
        vertexX = eventDict["vertexX"][iSourceParticle]
        vertexY = eventDict["vertexY"][iSourceParticle]
        vertexZ = eventDict["vertexZ"][iSourceParticle]
        trackEndX = eventDict["trackEndX"][iSourceParticle]
        trackEndY = eventDict["trackEndY"][iSourceParticle]
        trackEndZ = eventDict["trackEndZ"][iSourceParticle]
        showerDirX = eventDict["showerDirX"][iSourceParticle]
        showerDirY = eventDict["showerDirY"][iSourceParticle]
        showerDirZ = eventDict["showerDirZ"][iSourceParticle]
        ivysaurusMuon = eventDict["ivysaurusMuon"][iSourceParticle]
        ivysaurusProton = eventDict["ivysaurusProton"][iSourceParticle]
        ivysaurusPion = eventDict["ivysaurusPion"][iSourceParticle]
        ivysaurusElectron = eventDict["ivysaurusElectron"][iSourceParticle]
        ivysaurusPhoton = eventDict["ivysaurusPhoton"][iSourceParticle]
        trackLength = eventDict["trackLength"][iSourceParticle]
        displacement = eventDict["displacement"][iSourceParticle]
        dca = eventDict["dca"][iSourceParticle]
        isNeutrinoPDG = eventDict["isNeutrinoPDG"][iSourceParticle]
        nuVertexEnergyAsymmetry = eventDict["nuVertexEnergyAsymmetry"][iSourceParticle]
        nuVertexEnergyWeightedMeanRadialDistance = eventDict["nuVertexEnergyWeightedMeanRadialDistance"][iSourceParticle]
        
        # I don't know why, but sometimes the ivysaurus score is -inf - cry
        if (math.isnan(ivysaurusMuon) or math.isnan(ivysaurusProton) or math.isnan(ivysaurusPion) \
            or math.isnan(ivysaurusElectron) or math.isnan(ivysaurusPhoton)) :
            continue
            
        # Create our node features
        thisModeFeatures = [trackShowerScore, charge, vertexX, vertexY, vertexZ, \
                            showerDirX, showerDirY, showerDirZ, \
                            ivysaurusMuon, ivysaurusProton, \
                            ivysaurusPion, ivysaurusElectron, ivysaurusPhoton, \
                            displacement, dca, isNeutrinoPDG, trackLength, \
                            trackEndX, trackEndY, trackEndZ, nHits, \
                            nuVertexEnergyAsymmetry, nuVertexEnergyWeightedMeanRadialDistance]

        # Append our node
        nGraphNodes += 1
        event_node_features.append(thisModeFeatures)
        pfpIndex.append(iSourceParticle)
        
    # Can we actually create a graph?    
    if (nGraphNodes == 0) :
        print('Have no nodes!')
        # Bail if we have no graph nodes
        event_node_features = torch.tensor(event_node_features, dtype=torch.float)
        event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
        event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
        event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
        
        return Data(x=event_node_features, edge_index=event_edge_index_pos), \
        Data(x=event_node_features, edge_index=event_edge_index_neg), Data(x=event_node_features, edge_index=event_edge_index_FC)

        
    # This is to then add in particle-particle edges
    dtype = [('source', int), ('target', int), ('separationSq', float), ('angle', float), ('rank', int)]
    values = []
    
    # Create our NEUTRINO-PARTICLE edges
    for iSourceParticle in range(nGraphNodes) :
        
        # Get input vector index of source PFP 
        sourcePFPIndex = pfpIndex[iSourceParticle]
        
        for iTargetParticle in range((iSourceParticle + 1), nGraphNodes) :
            
            # Get input vector index of target PFP 
            targetPFPIndex = pfpIndex[iTargetParticle]
                
            # Demand that one of them is the neutrino
            isNeutrinoLink = (eventDict["isNeutrinoPDG"][targetPFPIndex] == 1) or (eventDict["isNeutrinoPDG"][sourcePFPIndex] == 1)
            
            if not isNeutrinoLink :
                # Save particle-particle knowledge, so we don't have to re-loop
                if modeDict["MAKE_PARTICLE_PARTICLE_LINKS"] :
                    # Calculate opening angle - do need to worry about normalisation
                    sourceDirection = np.array(showerDir_notNorm[sourcePFPIndex])
                    targetDirection = np.array(showerDir_notNorm[targetPFPIndex])
                    dotProduct = max(min(np.dot(sourceDirection, targetDirection), float(1.0)), float(-1.0))
                    openingAngle = np.arccos(dotProduct) * 180.0 / 3.14
                
                    # Calculate minimum separation - do need to worry about normalisation
                    positions1 = [vertex_notNorm[sourcePFPIndex], trackEnd_notNorm[sourcePFPIndex]]
                    positions2 = [vertex_notNorm[targetPFPIndex], trackEnd_notNorm[targetPFPIndex]]
                
                    minSeparationSq = sys.float_info.max
                
                    for position1 in positions1 :
                        for position2 in positions2 :
                            separationSq = np.linalg.norm(position1 - position2)
                        
                            if (float(separationSq) < minSeparationSq) :
                                minSeparationSq = separationSq
                
                    # Append to dictionary
                    values.append((iSourceParticle, iTargetParticle, minSeparationSq, openingAngle, 0))
                continue
                
            # Work out which index is which
            iNeutrino = iSourceParticle if (eventDict["isNeutrinoPDG"][sourcePFPIndex] == 1) else iTargetParticle
            iParticle = iTargetParticle if (eventDict["isNeutrinoPDG"][sourcePFPIndex] == 1) else iSourceParticle
     
            # Fill message passing network
            source_index_FC.append(iNeutrino)
            target_index_FC.append(iParticle)    
            
            ###################################################
            if modeDict["IS_TRAINING_MODE"] :
                # Is the edge correct?
                isTrueEdge = (eventDict["trueTrackID"][targetPFPIndex] == eventDict["trueVisibleParentTrackID"][sourcePFPIndex]) or \
                             (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex])
            
                # Append our edges, so that they're undirected
                # Only fill pos/neg with neutrino links
                if (isTrueEdge) :
                    # one way
                    source_index_pos.append(iNeutrino)
                    target_index_pos.append(iParticle)
                    # and back the other way
                    #source_index_pos.append(iParticle) 
                    #target_index_pos.append(iNeutrino)
                    #nPosEdge += 2
                    nPosEdge += 1
                else : 
                    # one way
                    source_index_neg.append(iSourceParticle) 
                    target_index_neg.append(iTargetParticle)
                    # and back the other way
                    source_index_neg.append(iTargetParticle)
                    target_index_neg.append(iSourceParticle)
                    nNegEdge += 2
            ###################################################
            
    # Can we make our training graphs?        
    if modeDict["IS_TRAINING_MODE"] :    
        if ((nPosEdge == 0) or (nNegEdge == 0)) :
            print("We cannot make our training graphs")
                  
            # Bail if we have no graph nodes
            event_node_features = torch.tensor(event_node_features, dtype=torch.float)
            event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
            event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
            event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
                  
            return Data(x=event_node_features, edge_index=event_edge_index_pos), \
        Data(x=event_node_features, edge_index=event_edge_index_neg), Data(x=event_node_features, edge_index=event_edge_index_FC)
        
    # Now make PARTICLE-PARTICLE matches
    if modeDict["MAKE_PARTICLE_PARTICLE_LINKS"] :
                  
        particle_particle_edges = np.array(values, dtype=dtype)       # create a structured array
    
        # Do the opening angle ranking
        particle_particle_edges = np.sort(particle_particle_edges, order='angle')    
        for index in range(len(particle_particle_edges)) :
            particle_particle_edges['rank'][index] += index

        # Do the separation ranking
        particle_particle_edges = np.sort(particle_particle_edges, order='separationSq') 
        for index in range(len(particle_particle_edges)) :
            particle_particle_edges['rank'][index] += index

        # Add in the edges to the message passing network
        particle_particle_edges = np.sort(particle_particle_edges, order='rank') 
        for index in range(math.floor(len(particle_particle_edges) * modeDict["EDGE_FRACTION"])) :
            # One way
            source_index_FC.append(particle_particle_edges['source'][index])
            target_index_FC.append(particle_particle_edges['target'][index])            
            # Other way 
            source_index_FC.append(particle_particle_edges['target'][index])
            target_index_FC.append(particle_particle_edges['source'][index])            
                
    event_node_features = torch.tensor(event_node_features, dtype=torch.float)
    event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
    event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
    event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
        
    # turn into data
    data_pos = Data(x=event_node_features, edge_index=event_edge_index_pos)
    data_neg = Data(x=event_node_features, edge_index=event_edge_index_neg)
    data_FC = Data(x=event_node_features, edge_index=event_edge_index_FC)
                  
    return data_pos, data_neg, data_FC
