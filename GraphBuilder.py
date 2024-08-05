import math
import numpy as np
import sys
import torch
from torch_geometric.data import Data
from tensorflow.keras.utils import to_categorical

import Normalisation

#############################
# Graph Builder Function
#############################
def GraphBuilder(eventDict, modeDict) :
        
    # Our lists that we want to fill
    event_node_features = []
    event_node_truth = []
    
    source_index_pos = []
    target_index_pos = []
    edge_is_primary_pos = []
    
    source_index_neg = []
    target_index_neg = []
    edge_is_primary_neg = []
    
    source_index_FC = []
    target_index_FC = []
    edge_weight_FC = [] # do dca?
    
    pfpIndex = []
    
    # Because i'm an idiot and set the vertex default value bad
    failedVertexMask = eventDict["vertexX"] < -990.0
    eventDict["vertexX"][failedVertexMask] = -9999.0
    eventDict["vertexY"][failedVertexMask] = -9999.0
    eventDict["vertexZ"][failedVertexMask] = -9999.0
    failedEndpointMask = eventDict["trackEndX"] < -990.0
    eventDict["trackEndX"][failedEndpointMask] = -9999.0
    eventDict["trackEndY"][failedEndpointMask] = -9999.0
    eventDict["trackEndZ"][failedEndpointMask] = -9999.0
    
    # I need to add in the neutrino 
    if modeDict["ADD_NEUTRINO"] :
        eventDict["nParticles"]                              += 1
        # For node
        eventDict["trackShowerScore"]                         = np.array(np.append(eventDict["trackShowerScore"], -999.0))
        eventDict["nHits"]                                    = np.array(np.append(eventDict["nHits"], -999.0))
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
        eventDict["isNeutrinoPDG"]                            = np.array(np.append(eventDict["isNeutrinoPDG"], int(1)), dtype='int8')
        eventDict["nuVertexEnergyAsymmetry"]                  = np.array(np.append(eventDict["nuVertexEnergyAsymmetry"], -999.0))
        eventDict["nuVertexEnergyWeightedMeanRadialDistance"] = np.array(np.append(eventDict["nuVertexEnergyWeightedMeanRadialDistance"], -999.0))
        # For training
        eventDict["trueTrackID"]                              = np.array(np.append(eventDict["trueTrackID"], 0))
        eventDict["truePDG"]                                  = np.array(np.append(eventDict["truePDG"], -1))
        eventDict["trueVisibleParentTrackID"]                 = np.array(np.append(eventDict["trueVisibleParentTrackID"], -999))
        eventDict["trueVisibleGeneration"]                    = np.array(np.append(eventDict["trueVisibleGeneration"], 1))
        
        # True information.. (for cheating)
        if modeDict["CHEAT_DIRECTION"] :
            eventDict["trueMomX"]                          = np.array(np.append(eventDict["trueMomX"], -999.0))
            eventDict["trueMomY"]                          = np.array(np.append(eventDict["trueMomY"], -999.0))
            eventDict["trueMomZ"]                          = np.array(np.append(eventDict["trueMomZ"], -999.0))
     
    # We need not normalised versions of some vectors... 
    vertex_notNorm = np.concatenate((eventDict["vertexX"].reshape(-1,1), eventDict["vertexY"].reshape(-1,1), eventDict["vertexZ"].reshape(-1,1)), axis=1)
    trackEnd_notNorm = np.concatenate((eventDict["trackEndX"].reshape(-1,1), eventDict["trackEndY"].reshape(-1,1), eventDict["trackEndZ"].reshape(-1,1)), axis=1)
    showerDir_notNorm = np.concatenate((eventDict["showerDirX"].reshape(-1,1), eventDict["showerDirY"].reshape(-1,1), eventDict["showerDirZ"].reshape(-1,1)), axis=1)
    dca_notNorm = eventDict["dca"]
    
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
        
        # Update the not normalised vector versions... 
        dca_notNorm = eventDict["dca"]
        
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
        # Node truth
        trueVisibleGeneration = eventDict["trueVisibleGeneration"][iSourceParticle]
        
        # Skip 2D particles?? - i think that this will do it...
        if (vertex_notNorm[iSourceParticle][0] < -990.0) :
            continue
        
        # I don't know why, but sometimes the ivysaurus score is -inf - cry
        if (math.isnan(ivysaurusMuon) or math.isnan(ivysaurusProton) or math.isnan(ivysaurusPion) \
            or math.isnan(ivysaurusElectron) or math.isnan(ivysaurusPhoton)) :
            print('skipping ivysaurus for index:', iSourceParticle)
            continue
            
        # Create our node features
        thisModeFeatures = [trackShowerScore, charge, vertexX, vertexY, vertexZ, \
                            showerDirX, showerDirY, showerDirZ, \
                            ivysaurusMuon, ivysaurusProton, \
                            ivysaurusPion, ivysaurusElectron, ivysaurusPhoton, \
                            displacement, dca, isNeutrinoPDG, trackLength, \
                            trackEndX, trackEndY, trackEndZ, nHits, \
                            nuVertexEnergyAsymmetry, nuVertexEnergyWeightedMeanRadialDistance]
        
        # Get node truth
        thisNodeTruth = (trueVisibleGeneration - 1) # So 0 = nu, 1 = primaries etc...
        thisNodeTruth = min(thisNodeTruth, modeDict["MAX_NODE_CLASS"]) # Tier class cap
        thisNodeTruth = modeDict["MAX_NODE_CLASS"] if (thisNodeTruth < 0) else thisNodeTruth # Targets -999 instances
        thisNodeTruth = to_categorical(thisNodeTruth, modeDict["MAX_NODE_CLASS"] + 1).tolist() # To include 0
        
        # Append our node
        nGraphNodes += 1
        event_node_features.append(thisModeFeatures)
        event_node_truth.append(thisNodeTruth)
        pfpIndex.append(iSourceParticle)
        
    # Can we actually create a graph?    
    if (nGraphNodes == 0) :
        # Bail if we have no graph nodes
        event_node_truth = torch.tensor(event_node_truth, dtype=torch.float)
        event_node_features = torch.tensor(event_node_features, dtype=torch.float)
        event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
        event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
        event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
        edge_is_primary_pos = torch.tensor(edge_is_primary_pos, dtype=torch.bool)
        edge_is_primary_neg = torch.tensor(edge_is_primary_neg, dtype=torch.bool)
        edge_weight_FC = torch.tensor(edge_weight_FC, dtype=torch.float)
        
        return Data(x=event_node_features, edge_index=event_edge_index_pos, y=event_node_truth, edge_attr=edge_is_primary_pos), \
        Data(x=event_node_features, edge_index=event_edge_index_neg, y=event_node_truth, edge_attr=edge_is_primary_neg), \
        Data(x=event_node_features, edge_index=event_edge_index_FC, y=event_node_truth, edge_attr=edge_weight_FC), \
        np.empty(0)

        
    # This is to then add in particle-particle edges
    dtype = [('source', int), ('target', int), ('separationSq', float), ('angle', float), ('rank', int)]
    values = []
    
    # Create our NU-PARTICLE edges and POS-NEG graphs
    for iSourceParticle in range(nGraphNodes) :
        
        # Get input vector index of source PFP 
        sourcePFPIndex = pfpIndex[iSourceParticle]
        
        for iTargetParticle in range((iSourceParticle + 1), nGraphNodes) :
            
            # Get input vector index of target PFP 
            targetPFPIndex = pfpIndex[iTargetParticle]
              
            # Is this a true edge?
            isTrueEdge = (eventDict["trueTrackID"][targetPFPIndex] == eventDict["trueVisibleParentTrackID"][sourcePFPIndex]) or \
                         (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex])
                
            # Is one of them the neutrino?
            isNeutrinoLink = (eventDict["isNeutrinoPDG"][targetPFPIndex] == 1) or (eventDict["isNeutrinoPDG"][sourcePFPIndex] == 1)
            
            # Work out which index is which
            iParent = iSourceParticle if (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex]) else iTargetParticle
            iChild = iTargetParticle if (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex]) else iSourceParticle
            
            parentIndex = sourcePFPIndex if (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex]) else targetPFPIndex
            targetIndex = targetPFPIndex if (eventDict["trueTrackID"][sourcePFPIndex] == eventDict["trueVisibleParentTrackID"][targetPFPIndex]) else sourcePFPIndex
     
            # Fill pos/neg graphs 
            if (isNeutrinoLink) :
                
                # 100 hits
                passHitThreshold = (eventDict["nHits"][targetPFPIndex] > 0.05) or (eventDict["nHits"][sourcePFPIndex] > 0.05)
                
                if (modeDict["IS_PRIMARY_TRAINING"] and passHitThreshold) :
                    # Append our edges, so that they're directed? why not.
                    # Only fill pos/neg with neutrino links
                    if (isTrueEdge) :
                        # one way
                        source_index_pos.append(iParent)
                        target_index_pos.append(iChild)
                        edge_is_primary_pos.append([True])
                        # and back the other way
                        source_index_pos.append(iChild) 
                        target_index_pos.append(iParent)
                        edge_is_primary_pos.append([True])
                        # increase edge count
                        nPosEdge += 2
                    else : 
                        # one way
                        source_index_neg.append(iParent) 
                        target_index_neg.append(iChild)
                        edge_is_primary_neg.append([True])
                        # and back the other way
                        source_index_neg.append(iChild)
                        target_index_neg.append(iParent)
                        edge_is_primary_neg.append([True])
                        # increase edge count
                        nNegEdge += 2

#             else :
#                 if modeDict["IS_HIGHER_TIER_TRAINING"] :
                    
#                     # 100 hits
#                     passHitThreshold = (eventDict["nHits"][targetPFPIndex] > 0.05) and (eventDict["nHits"][sourcePFPIndex] > 0.05)
                    
#                     # Ignore true primary targets, i.e. do not make primary-primary edges
#                     #isTargetTruePrimary = (eventDict["trueVisibleParentTrackID"][targetPFPIndex] == 0)
                    
#                     #if not isTargetTruePrimary :
#                     if (isTrueEdge) :
#                         # one way
#                         source_index_pos.append(iParent)
#                         target_index_pos.append(iChild)
#                         edge_is_primary_pos.append([False])
#                         # and back the other way
#                         source_index_pos.append(iChild) 
#                         target_index_pos.append(iParent)
#                         edge_is_primary_pos.append([False])
#                         # increase edge count
#                         nPosEdge += 2
#                     else : 
#                         # one way
#                         source_index_neg.append(iParent) 
#                         target_index_neg.append(iChild)
#                         edge_is_primary_neg.append([False])
#                         # and back the other way
#                         source_index_neg.append(iChild)
#                         target_index_neg.append(iParent)
#                         edge_is_primary_neg.append([False])
#                         # increase edge count
#                         nNegEdge += 2
            
            # Fill message passing graph info
            if isNeutrinoLink :
                # Add one way
                source_index_FC.append(iParent)
                target_index_FC.append(iChild)
                # Add the edge weight - smaller DCA is more important
                edge_weight_FC.append([0.0 if (dca_notNorm[targetIndex] < 0.0001) else 1.0 / dca_notNorm[targetIndex]])
                #edge_is_primary_FC.append([True])
            else :
                # Need to save this info, so I can do ranking
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
            
    # Can we make our training graphs?        
    if ((modeDict["IS_PRIMARY_TRAINING"]) or (modeDict["IS_HIGHER_TIER_TRAINING"])) :    
        if ((nPosEdge == 0) and (nNegEdge == 0)) :       
            # Bail if we have no graph edges
            event_node_truth = torch.tensor(event_node_truth, dtype=torch.float)
            event_node_features = torch.tensor(event_node_features, dtype=torch.float)
            event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
            event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
            event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
            edge_is_primary_pos = torch.tensor(edge_is_primary_pos, dtype=torch.bool)
            edge_is_primary_neg = torch.tensor(edge_is_primary_neg, dtype=torch.bool)
            edge_weight_FC = torch.tensor(edge_weight_FC, dtype=torch.float)
                
            return Data(x=event_node_features, edge_index=event_edge_index_pos, y=event_node_truth, edge_attr=edge_is_primary_pos), \
            Data(x=event_node_features, edge_index=event_edge_index_neg, y=event_node_truth, edge_attr=edge_is_primary_neg), \
            Data(x=event_node_features, edge_index=event_edge_index_FC, y=event_node_truth, edge_attr=edge_weight_FC), \
            np.empty(0)
        
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
            #edge_is_primary_FC.append([False])
            # Add the edge weight - smaller DCA is more important
            edge_weight_FC.append([1.0])
            # Other way 
            source_index_FC.append(particle_particle_edges['target'][index])
            target_index_FC.append(particle_particle_edges['source'][index])        
            #edge_is_primary_FC.append([False])
            # Add the edge weight - smaller DCA is more important
            edge_weight_FC.append([1.0])
                    
    #print('event_truth_pos:', event_truth_pos)
    #print('event_truth_neg:', event_truth_neg)
    #print('event_truth_FC:', event_truth_FC)
    
    event_node_truth = torch.tensor(event_node_truth, dtype=torch.float)
    event_node_features = torch.tensor(event_node_features, dtype=torch.float)
    event_edge_index_pos = torch.tensor([source_index_pos, target_index_pos], dtype=torch.long)
    event_edge_index_neg = torch.tensor([source_index_neg, target_index_neg], dtype=torch.long)
    event_edge_index_FC = torch.tensor([source_index_FC, target_index_FC], dtype=torch.long)
    edge_is_primary_pos = torch.tensor(edge_is_primary_pos, dtype=torch.bool)
    edge_is_primary_neg = torch.tensor(edge_is_primary_neg, dtype=torch.bool)
    edge_weight_FC = torch.tensor(edge_weight_FC, dtype=torch.float)
    
    # turn into data
    data_pos = Data(x=event_node_features, edge_index=event_edge_index_pos, y=event_node_truth, edge_attr=edge_is_primary_pos)
    data_neg = Data(x=event_node_features, edge_index=event_edge_index_neg, y=event_node_truth, edge_attr=edge_is_primary_neg)
    data_FC = Data(x=event_node_features, edge_index=event_edge_index_FC, y=event_node_truth, edge_attr=edge_weight_FC)
                  
    return data_pos, data_neg, data_FC, np.array(pfpIndex)
