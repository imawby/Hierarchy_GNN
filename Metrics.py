import numpy as np
import matplotlib.pyplot as plt

BOGUS_INT = -999

####################################################################################################################################### 
#######################################################################################################################################

def calculateMetrics_leading_lepton(nEntries, particleMask_in, nSpacepoints_in, pfpTruePDG_in, trueVisibleGeneration_in, isNC_in, nuPDG_in, new_gen) :

    n_two_d = 0
    n_true_muon = 0
    n_true_electron = 0
    n_true_pion = 0
    n_true_photon = 0
    n_true_proton = 0

    # metrics
    n_tagged_as_primary_muon = 0
    n_incorrect_parent_muon = 0
    n_not_tagged_muon = 0 

    n_tagged_as_primary_electron = 0
    n_incorrect_parent_electron = 0
    n_not_tagged_electron = 0 
    
    n_tagged_as_primary_pion = 0
    n_incorrect_parent_pion = 0
    n_not_tagged_pion = 0 
    
    n_tagged_as_primary_photon = 0
    n_incorrect_parent_photon = 0
    n_not_tagged_photon = 0 
    
    n_tagged_as_primary_proton = 0
    n_incorrect_parent_proton = 0
    n_not_tagged_proton = 0 

    for iEvent in range(nEntries) : 

#         # if target event?
#         if (isNC_in[iEvent] == 1) :
#             continue            
            
#         if ((nuPDG_in[iEvent] != 12) and (nuPDG_in[iEvent] != 14)) :
#             continue
            
        # Particle mask
        particle_mask = np.array(particleMask_in[iEvent])
        # PFP info
        nSpacepoints_np = np.array(nSpacepoints_in[iEvent])[particle_mask]
        # Truth
        trueVisibleGeneration_np = np.array(trueVisibleGeneration_in[iEvent])[particle_mask]
        pfpTruePDG_np = np.array(pfpTruePDG_in[iEvent])[particle_mask]
        # New
        newGen_np = np.array(new_gen[iEvent])[particle_mask]

        #########################
        # Get tier masks
        #########################
        ignore_mask = np.logical_or(nSpacepoints_np == 0, trueVisibleGeneration_np == -999)
        target_mask = np.logical_not(ignore_mask)
        true_primary_mask = np.logical_and(target_mask, trueVisibleGeneration_np == 2)
        target_muon_mask = np.logical_and(true_primary_mask, np.abs(pfpTruePDG_np) == 13)
        target_proton_mask = np.logical_and(true_primary_mask, np.abs(pfpTruePDG_np) == 2212)
        target_pion_mask = np.logical_and(true_primary_mask, np.abs(pfpTruePDG_np) == 211)
        target_electron_mask = np.logical_and(true_primary_mask, np.abs(pfpTruePDG_np) == 11)
        target_photon_mask = np.logical_and(true_primary_mask, np.abs(pfpTruePDG_np) == 22)

        # metrics
        n_true_muon += np.count_nonzero(target_muon_mask)
        n_true_electron += np.count_nonzero(target_electron_mask)
        n_true_pion += np.count_nonzero(target_pion_mask)
        n_true_photon += np.count_nonzero(target_photon_mask)
        n_true_proton += np.count_nonzero(target_proton_mask)
        
        n_tagged_as_primary_muon += np.count_nonzero(newGen_np[target_muon_mask] == 2)
        n_incorrect_parent_muon += np.count_nonzero(np.logical_and(newGen_np[target_muon_mask] != 2, newGen_np[target_muon_mask] != BOGUS_INT))
        n_not_tagged_muon += np.count_nonzero(newGen_np[target_muon_mask] == BOGUS_INT)

        n_tagged_as_primary_electron += np.count_nonzero(newGen_np[target_electron_mask] == 2)
        n_incorrect_parent_electron += np.count_nonzero(np.logical_and(newGen_np[target_electron_mask] != 2, newGen_np[target_electron_mask] != BOGUS_INT))
        n_not_tagged_electron += np.count_nonzero(newGen_np[target_electron_mask] == BOGUS_INT)
        
        n_tagged_as_primary_proton += np.count_nonzero(newGen_np[target_proton_mask] == 2)
        n_incorrect_parent_proton += np.count_nonzero(np.logical_and(newGen_np[target_proton_mask] != 2, newGen_np[target_proton_mask] != BOGUS_INT))
        n_not_tagged_proton += np.count_nonzero(newGen_np[target_proton_mask] == BOGUS_INT)
        
        n_tagged_as_primary_pion += np.count_nonzero(newGen_np[target_pion_mask] == 2)
        n_incorrect_parent_pion += np.count_nonzero(np.logical_and(newGen_np[target_pion_mask] != 2, newGen_np[target_pion_mask] != BOGUS_INT))
        n_not_tagged_pion += np.count_nonzero(newGen_np[target_pion_mask] == BOGUS_INT)
        
        n_tagged_as_primary_photon += np.count_nonzero(newGen_np[target_photon_mask] == 2)
        n_incorrect_parent_photon += np.count_nonzero(np.logical_and(newGen_np[target_photon_mask] != 2, newGen_np[target_photon_mask] != BOGUS_INT))
        n_not_tagged_photon += np.count_nonzero(newGen_np[target_photon_mask] == BOGUS_INT)
        
    #############################################
    # Calc fraction
    #############################################   

    n_tagged_as_primary_muon_frac = round(0.0 if n_true_muon == 0 else float(n_tagged_as_primary_muon) / float(n_true_muon), 2)
    n_incorrect_parent_muon_frac = round(0.0 if n_true_muon == 0 else float(n_incorrect_parent_muon) / float(n_true_muon), 2)
    n_not_tagged_muon_frac = round(0.0 if n_true_muon == 0 else float(n_not_tagged_muon) / float(n_true_muon), 2)

    n_tagged_as_primary_electron_frac = round(0.0 if n_true_electron == 0 else float(n_tagged_as_primary_electron) / float(n_true_electron), 2)
    n_incorrect_parent_electron_frac = round(0.0 if n_true_electron == 0 else float(n_incorrect_parent_electron) / float(n_true_electron), 2)
    n_not_tagged_electron_frac = round(0.0 if n_true_electron == 0 else float(n_not_tagged_electron) / float(n_true_electron), 2)
    
    n_tagged_as_primary_proton_frac = round(0.0 if n_true_proton == 0 else float(n_tagged_as_primary_proton) / float(n_true_proton), 2)
    n_incorrect_parent_proton_frac = round(0.0 if n_true_proton == 0 else float(n_incorrect_parent_proton) / float(n_true_proton), 2)
    n_not_tagged_proton_frac = round(0.0 if n_true_proton == 0 else float(n_not_tagged_proton) / float(n_true_proton), 2)
    
    n_tagged_as_primary_pion_frac = round(0.0 if n_true_pion == 0 else float(n_tagged_as_primary_pion) / float(n_true_pion), 2)
    n_incorrect_parent_pion_frac = round(0.0 if n_true_pion == 0 else float(n_incorrect_parent_pion) / float(n_true_pion), 2)
    n_not_tagged_pion_frac = round(0.0 if n_true_pion == 0 else float(n_not_tagged_pion) / float(n_true_pion), 2)
    
    n_tagged_as_primary_photon_frac = round(0.0 if n_true_photon == 0 else float(n_tagged_as_primary_photon) / float(n_true_photon), 2)
    n_incorrect_parent_photon_frac = round(0.0 if n_true_photon == 0 else float(n_incorrect_parent_photon) / float(n_true_photon), 2)
    n_not_tagged_photon_frac = round(0.0 if n_true_photon == 0 else float(n_not_tagged_photon) / float(n_true_photon), 2)
    
    print('')
    print('-------------------------------------------------------------')
    print('NEW - True Gen   | Muon | Electron | Photon | Pion | Proton |')
    print('-------------------------------------------------------------')
    print('Correct primary  |' + str(n_tagged_as_primary_muon_frac) + str(' '* (6 - len(str(n_tagged_as_primary_muon_frac)))) + \
                           '|' + str(n_tagged_as_primary_electron_frac) + str(' '* (10 - len(str(n_tagged_as_primary_electron_frac)))) + \
                           '|' + str(n_tagged_as_primary_proton_frac) + str(' '* (8 - len(str(n_tagged_as_primary_proton_frac)))) + \
                           '|' + str(n_tagged_as_primary_pion_frac) + str(' '* (6 - len(str(n_tagged_as_primary_pion_frac)))) + \
                           '|' + str(n_tagged_as_primary_photon_frac) + str(' '* (8 - len(str(n_tagged_as_primary_photon_frac)))) + \
                           '|')
    print('Incorrect parent |' + str(n_incorrect_parent_muon_frac) + str(' '* (6 - len(str(n_incorrect_parent_muon_frac)))) + \
                           '|' + str(n_incorrect_parent_electron_frac) + str(' '* (10 - len(str(n_incorrect_parent_electron_frac)))) + \
                           '|' + str(n_incorrect_parent_proton_frac) + str(' '* (8 - len(str(n_incorrect_parent_proton_frac)))) + \
                           '|' + str(n_incorrect_parent_pion_frac) + str(' '* (6 - len(str(n_incorrect_parent_pion_frac)))) + \
                           '|' + str(n_incorrect_parent_photon_frac) + str(' '* (8 - len(str(n_incorrect_parent_photon_frac)))) + \
                           '|')
    print('Not tagged       |' + str(n_not_tagged_muon_frac) + str(' '* (6 - len(str(n_not_tagged_muon_frac)))) + \
                           '|' + str(n_not_tagged_electron_frac) + str(' '* (10 - len(str(n_not_tagged_electron_frac)))) + \
                           '|' + str(n_not_tagged_proton_frac) + str(' '* (8 - len(str(n_not_tagged_proton_frac)))) + \
                           '|' + str(n_not_tagged_pion_frac) + str(' '* (6 - len(str(n_not_tagged_pion_frac)))) + \
                           '|' + str(n_not_tagged_photon_frac) + str(' '* (8 - len(str(n_not_tagged_photon_frac)))) + \
                           '|')
    print('-------------------------------------------------------------')
    print('Total             |' + str(n_true_muon) + str(' '* (6 - len(str(n_true_muon)))) + \
                            '|' + str(n_true_electron) + str(' '* (10 - len(str(n_true_electron)))) + \
                            '|' + str(n_true_proton) + str(' '* (8 - len(str(n_true_proton)))) + \
                            '|' + str(n_true_pion) + str(' '* (6 - len(str(n_true_pion)))) + \
                            '|' + str(n_true_photon) + str(' '* (8 - len(str(n_true_photon)))) + \
                            '|')
    print('-------------------------------------------------------------')
    print('')

####################################################################################################################################### 
#######################################################################################################################################

def calculateMetrics_new(nEntries, particleMask_in, nSpacepoints_in, trackShowerScore_in, trueVisibleGeneration_in, trueVisibleParentPFPIndex_in, new_parentPFPIndex, new_gen) :
    for isTrack in [True, False] :
        n_two_d = 0
        n_true_primary = 0
        n_true_secondary = 0
        n_true_tertiary = 0
        n_true_higher = 0

        # NEW!
        n_correct_parent_correct_tier_primary = 0
        n_correct_parent_wrong_tier_primary = 0
        n_tagged_as_primary_primary = 0
        n_incorrect_parent_primary = 0
        n_not_tagged_primary = 0 

        n_correct_parent_correct_tier_secondary = 0
        n_correct_parent_wrong_tier_secondary = 0
        n_tagged_as_primary_secondary = 0
        n_incorrect_parent_secondary = 0
        n_not_tagged_secondary = 0 

        n_correct_parent_correct_tier_tertiary = 0
        n_correct_parent_wrong_tier_tertiary = 0
        n_correct_parent_tertiary = 0
        n_tagged_as_primary_tertiary = 0
        n_incorrect_parent_tertiary = 0
        n_not_tagged_tertiary = 0 

        n_correct_parent_correct_tier_higher = 0
        n_correct_parent_wrong_tier_higher = 0
        n_correct_parent_higher = 0
        n_tagged_as_primary_higher = 0
        n_incorrect_parent_higher = 0
        n_not_tagged_higher = 0 

        for iEvent in range(nEntries) : 

            # Particle mask
            particle_mask = np.array(particleMask_in[iEvent])
            # PFP info
            nSpacepoints_np = np.array(nSpacepoints_in[iEvent])[particle_mask]
            trackShowerScore_np = np.array(trackShowerScore_in[iEvent])[particle_mask]
            # Truth
            trueVisibleGeneration_np = np.array(trueVisibleGeneration_in[iEvent])[particle_mask]
            trueVisibleParentPFPIndex_np = np.array(trueVisibleParentPFPIndex_in[iEvent])[particle_mask]
            # New
            newParentPFPIndex_np = np.array(new_parentPFPIndex[iEvent])[particle_mask]
            newGen_np = np.array(new_gen[iEvent])[particle_mask]

            #########################
            # Get tier masks
            #########################
            ignore_mask = np.logical_or(nSpacepoints_np == 0, trueVisibleGeneration_np == -999)
            trackShower_mask = (trackShowerScore_np > 0.5) if isTrack else np.logical_not(trackShowerScore_np > 0.5)
            true_primary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 2)
            true_secondary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 3)
            true_tertiary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 4)
            true_higher_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), 
                                              np.logical_not(np.logical_or(true_primary_mask, np.logical_or(true_secondary_mask, true_tertiary_mask))))

            #############################################
            # Get metrics for this event - debugging
            #############################################
            # Totals
            this_two_d = np.count_nonzero(np.logical_and(ignore_mask, trackShower_mask))
            this_true_primary = np.count_nonzero(true_primary_mask)
            this_true_secondary = np.count_nonzero(true_secondary_mask)
            this_true_tertiary = np.count_nonzero(true_tertiary_mask)
            this_true_higher = np.count_nonzero(true_higher_mask)

            # Primary
            this_correct_parent_correct_tier_primary = np.count_nonzero(newGen_np[true_primary_mask] == 2)
            this_correct_parent_wrong_tier_primary = 0
            this_tagged_as_primary_primary = 0
            this_not_tagged_primary = np.count_nonzero(newGen_np[true_primary_mask] == BOGUS_INT)
            this_incorrect_parent_primary = np.count_nonzero(np.logical_and(newGen_np[true_primary_mask] != 2, \
                                                                                newGen_np[true_primary_mask] != BOGUS_INT)) 
            # Secondary
            this_correct_parent_correct_tier_secondary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                         newGen_np[true_secondary_mask] == 3))
            this_correct_parent_wrong_tier_secondary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                       np.logical_and(newGen_np[true_secondary_mask] != 3, \
                                                                                                      newGen_np[true_secondary_mask] != BOGUS_INT)))
            this_tagged_as_primary_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == 2)
            this_not_tagged_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == BOGUS_INT)
            this_incorrect_parent_secondary = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                            np.logical_or(newGen_np[true_secondary_mask] == 2, \
                                                                                                          newGen_np[true_secondary_mask] == BOGUS_INT))))
            # Tertiary
            this_correct_parent_correct_tier_tertiary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                        newGen_np[true_tertiary_mask] == 4))
            this_correct_parent_wrong_tier_tertiary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                      np.logical_and(newGen_np[true_tertiary_mask] != 4, \
                                                                                                     newGen_np[true_tertiary_mask] != BOGUS_INT)))
            this_tagged_as_primary_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == 2)
            this_not_tagged_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == BOGUS_INT)
            this_incorrect_parent_tertiary = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                           np.logical_or(newGen_np[true_tertiary_mask] == 2, \
                                                                                                         newGen_np[true_tertiary_mask] == BOGUS_INT))))
            # Higher
            this_correct_parent_correct_tier_higher = 0
            this_correct_parent_wrong_tier_higher = np.count_nonzero(newParentPFPIndex_np[true_higher_mask] == trueVisibleParentPFPIndex_np[true_higher_mask])
            this_tagged_as_primary_higher = np.count_nonzero(newGen_np[true_higher_mask] == 2)
            this_not_tagged_higher = np.count_nonzero(newGen_np[true_higher_mask] == BOGUS_INT)
            this_incorrect_parent_higher = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_higher_mask] == trueVisibleParentPFPIndex_np[true_higher_mask], \
                                                                                         np.logical_or(newGen_np[true_higher_mask] == 2, \
                                                                                                       newGen_np[true_higher_mask] == BOGUS_INT))))


            #############################################
            # Add metrics to global
            #############################################
            n_two_d += this_two_d
            n_true_primary += this_true_primary
            n_true_secondary += this_true_secondary
            n_true_tertiary += this_true_tertiary
            n_true_higher += this_true_higher

            n_correct_parent_correct_tier_primary += this_correct_parent_correct_tier_primary
            n_correct_parent_wrong_tier_primary += this_correct_parent_wrong_tier_primary
            n_tagged_as_primary_primary += this_tagged_as_primary_primary
            n_incorrect_parent_primary += this_incorrect_parent_primary
            n_not_tagged_primary += this_not_tagged_primary
            n_correct_parent_correct_tier_secondary += this_correct_parent_correct_tier_secondary
            n_correct_parent_wrong_tier_secondary += this_correct_parent_wrong_tier_secondary
            n_tagged_as_primary_secondary += this_tagged_as_primary_secondary
            n_incorrect_parent_secondary += this_incorrect_parent_secondary
            n_not_tagged_secondary += this_not_tagged_secondary
            n_correct_parent_correct_tier_tertiary += this_correct_parent_correct_tier_tertiary
            n_correct_parent_wrong_tier_tertiary += this_correct_parent_wrong_tier_tertiary
            n_tagged_as_primary_tertiary += this_tagged_as_primary_tertiary
            n_incorrect_parent_tertiary += this_incorrect_parent_tertiary
            n_not_tagged_tertiary += this_not_tagged_tertiary
            n_correct_parent_correct_tier_higher += this_correct_parent_correct_tier_higher
            n_correct_parent_wrong_tier_higher += this_correct_parent_wrong_tier_higher
            n_tagged_as_primary_higher += this_tagged_as_primary_higher
            n_incorrect_parent_higher += this_incorrect_parent_higher
            n_not_tagged_higher += this_not_tagged_higher

        #############################################
        # Print metrics
        #############################################   

        n_correct_parent_correct_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_correct_tier_primary) / float(n_true_primary), 2)
        n_correct_parent_wrong_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_wrong_tier_primary) / float(n_true_primary), 2)
        n_tagged_as_primary_primary_frac = round(0.0 if n_true_primary == 0 else float(n_tagged_as_primary_primary) / float(n_true_primary), 2)
        n_incorrect_parent_primary_frac = round(0.0 if n_true_primary == 0 else float(n_incorrect_parent_primary) / float(n_true_primary), 2)
        n_not_tagged_primary_frac = round(0.0 if n_true_primary == 0 else float(n_not_tagged_primary) / float(n_true_primary), 2)

        n_correct_parent_correct_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_correct_tier_secondary) / float(n_true_secondary), 2)
        n_correct_parent_wrong_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_wrong_tier_secondary) / float(n_true_secondary), 2)
        n_tagged_as_primary_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_tagged_as_primary_secondary) / float(n_true_secondary), 2)
        n_incorrect_parent_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_incorrect_parent_secondary) / float(n_true_secondary), 2)
        n_not_tagged_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_not_tagged_secondary) / float(n_true_secondary), 2)

        n_correct_parent_correct_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_correct_tier_tertiary) / float(n_true_tertiary), 2)
        n_correct_parent_wrong_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_wrong_tier_tertiary) / float(n_true_tertiary), 2)
        n_tagged_as_primary_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_tagged_as_primary_tertiary) / float(n_true_tertiary), 2)
        n_incorrect_parent_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_incorrect_parent_tertiary) / float(n_true_tertiary), 2)
        n_not_tagged_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_not_tagged_tertiary) / float(n_true_tertiary), 2)

        n_correct_parent_correct_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_correct_tier_higher) / float(n_true_higher), 2)
        n_correct_parent_wrong_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_wrong_tier_higher) / float(n_true_higher), 2)
        n_tagged_as_primary_higher_frac = round(0.0 if n_true_higher == 0 else float(n_tagged_as_primary_higher) / float(n_true_higher), 2)
        n_incorrect_parent_higher_frac = round(0.0 if n_true_higher == 0 else float(n_incorrect_parent_higher) / float(n_true_higher), 2)
        n_not_tagged_higher_frac = round(0.0 if n_true_higher == 0 else float(n_not_tagged_higher) / float(n_true_higher), 2)


        print('------------------------------------------------------------')
        print(('TRACK' if isTrack else 'SHOWER'))
        print('------------------------------------------------------------')
        print('NEW - True Gen   | Primary | Secondary | Tertiary | Higher |')
        print('------------------------------------------------------------')
        print('Correct parent CT |' + str(n_correct_parent_correct_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_correct_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_correct_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_correct_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_correct_tier_higher_frac)))) + \
                                '|')
        print('Correct parent WT |' + str(n_correct_parent_wrong_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_wrong_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_wrong_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_wrong_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_wrong_tier_higher_frac)))) + \
                                '|')
        print('False primary     |' + str(n_tagged_as_primary_primary_frac) + str(' '* (9 - len(str(n_tagged_as_primary_primary_frac)))) + \
                                '|' + str(n_tagged_as_primary_secondary_frac) + str(' '* (11 - len(str(n_tagged_as_primary_secondary_frac)))) + \
                                '|' + str(n_tagged_as_primary_tertiary_frac) + str(' '* (10 - len(str(n_tagged_as_primary_tertiary_frac)))) + \
                                '|' + str(n_tagged_as_primary_higher_frac) + str(' '* (8 - len(str(n_tagged_as_primary_higher_frac)))) + \
                                '|')
        print('Incorrect parent  |' + str(n_incorrect_parent_primary_frac) + str(' '* (9 - len(str(n_incorrect_parent_primary_frac)))) + \
                                '|' + str(n_incorrect_parent_secondary_frac) + str(' '* (11 - len(str(n_incorrect_parent_secondary_frac)))) + \
                                '|' + str(n_incorrect_parent_tertiary_frac) + str(' '* (10 - len(str(n_incorrect_parent_tertiary_frac)))) + \
                                '|' + str(n_incorrect_parent_higher_frac) + str(' '* (8 - len(str(n_incorrect_parent_higher_frac)))) + \
                                '|')
        print('Not tagged        |' + str(n_not_tagged_primary_frac) + str(' '* (9 - len(str(n_not_tagged_primary_frac)))) + \
                                '|' + str(n_not_tagged_secondary_frac) + str(' '* (11 - len(str(n_not_tagged_secondary_frac)))) + \
                                '|' + str(n_not_tagged_tertiary_frac) + str(' '* (10 - len(str(n_not_tagged_tertiary_frac)))) + \
                                '|' + str(n_not_tagged_higher_frac) + str(' '* (8 - len(str(n_not_tagged_higher_frac)))) + \
                                '|')
        print('------------------------------------------------------------')
        print('Total             |' + str(n_true_primary) + str(' '* (9 - len(str(n_true_primary)))) + \
                                '|' + str(n_true_secondary) + str(' '* (11 - len(str(n_true_secondary)))) + \
                                '|' + str(n_true_tertiary) + str(' '* (10 - len(str(n_true_tertiary)))) + \
                                '|' + str(n_true_higher) + str(' '* (8 - len(str(n_true_higher)))) + \
                                '|')
        print('n_two_d (not included in metrics):', n_two_d)
        print('------------------------------------------------------------')
        print('')
     
     
####################################################################################################################################### 
#######################################################################################################################################

def calculateMetrics_new_flavour(nEntries, particleMask_in, nSpacepoints_in, trackShowerScore_in, trueVisibleGeneration_in, trueVisibleParentPFPIndex_in, isNC_in, nuPDG_in, new_parentPFPIndex, new_gen, target_NC, target_nuPDG) :
    for isTrack in [True, False] :
        n_two_d = 0
        n_true_primary = 0
        n_true_secondary = 0
        n_true_tertiary = 0
        n_true_higher = 0

        # NEW!
        n_correct_parent_correct_tier_primary = 0
        n_correct_parent_wrong_tier_primary = 0
        n_tagged_as_primary_primary = 0
        n_incorrect_parent_primary = 0
        n_not_tagged_primary = 0 

        n_correct_parent_correct_tier_secondary = 0
        n_correct_parent_wrong_tier_secondary = 0
        n_tagged_as_primary_secondary = 0
        n_incorrect_parent_secondary = 0
        n_not_tagged_secondary = 0 

        n_correct_parent_correct_tier_tertiary = 0
        n_correct_parent_wrong_tier_tertiary = 0
        n_correct_parent_tertiary = 0
        n_tagged_as_primary_tertiary = 0
        n_incorrect_parent_tertiary = 0
        n_not_tagged_tertiary = 0 

        n_correct_parent_correct_tier_higher = 0
        n_correct_parent_wrong_tier_higher = 0
        n_correct_parent_higher = 0
        n_tagged_as_primary_higher = 0
        n_incorrect_parent_higher = 0
        n_not_tagged_higher = 0 

        for iEvent in range(nEntries) : 

            # if target event?
            if ((isNC_in[iEvent] != target_NC) or (np.abs(nuPDG_in[iEvent]) != target_nuPDG)) :
                continue            
            
            # Particle mask
            particle_mask = np.array(particleMask_in[iEvent])
            # PFP info
            nSpacepoints_np = np.array(nSpacepoints_in[iEvent])[particle_mask]
            trackShowerScore_np = np.array(trackShowerScore_in[iEvent])[particle_mask]
            # Truth
            trueVisibleGeneration_np = np.array(trueVisibleGeneration_in[iEvent])[particle_mask]
            trueVisibleParentPFPIndex_np = np.array(trueVisibleParentPFPIndex_in[iEvent])[particle_mask]
            # New
            newParentPFPIndex_np = np.array(new_parentPFPIndex[iEvent])
            newGen_np = np.array(new_gen[iEvent])

            #########################
            # Get tier masks
            #########################
            ignore_mask = np.logical_or(nSpacepoints_np == 0, trueVisibleGeneration_np == -999)
            trackShower_mask = (trackShowerScore_np > 0.5) if isTrack else np.logical_not(trackShowerScore_np > 0.5)
            target_mask = np.logical_and(np.logical_not(ignore_mask), trackShower_mask)
            true_primary_mask = np.logical_and(target_mask, trueVisibleGeneration_np == 2)
            true_secondary_mask = np.logical_and(target_mask, trueVisibleGeneration_np == 3)
            true_tertiary_mask = np.logical_and(target_mask, trueVisibleGeneration_np == 4)
            true_higher_mask = np.logical_and(target_mask, np.logical_not(np.logical_or(true_primary_mask, np.logical_or(true_secondary_mask, true_tertiary_mask))))

            #############################################
            # Get metrics for this event - debugging
            #############################################
            # Totals
            this_two_d = np.count_nonzero(np.logical_and(ignore_mask, trackShower_mask))
            this_true_primary = np.count_nonzero(true_primary_mask)
            this_true_secondary = np.count_nonzero(true_secondary_mask)
            this_true_tertiary = np.count_nonzero(true_tertiary_mask)
            this_true_higher = np.count_nonzero(true_higher_mask)

            # Primary
            this_correct_parent_correct_tier_primary = np.count_nonzero(newGen_np[true_primary_mask] == 2)
            this_correct_parent_wrong_tier_primary = 0
            this_tagged_as_primary_primary = 0
            this_not_tagged_primary = np.count_nonzero(newGen_np[true_primary_mask] == BOGUS_INT)
            this_incorrect_parent_primary = np.count_nonzero(np.logical_and(newGen_np[true_primary_mask] != 2, \
                                                                                newGen_np[true_primary_mask] != BOGUS_INT)) 
            # Secondary
            this_correct_parent_correct_tier_secondary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                         newGen_np[true_secondary_mask] == 3))
            this_correct_parent_wrong_tier_secondary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                       np.logical_and(newGen_np[true_secondary_mask] != 3, \
                                                                                                      newGen_np[true_secondary_mask] != BOGUS_INT)))
            this_tagged_as_primary_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == 2)
            this_not_tagged_secondary = np.count_nonzero(newGen_np[true_secondary_mask] == BOGUS_INT)
            this_incorrect_parent_secondary = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_secondary_mask] == trueVisibleParentPFPIndex_np[true_secondary_mask], \
                                                                                            np.logical_or(newGen_np[true_secondary_mask] == 2, \
                                                                                                          newGen_np[true_secondary_mask] == BOGUS_INT))))
            # Tertiary
            this_correct_parent_correct_tier_tertiary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                        newGen_np[true_tertiary_mask] == 4))
            this_correct_parent_wrong_tier_tertiary = np.count_nonzero(np.logical_and(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                      np.logical_and(newGen_np[true_tertiary_mask] != 4, \
                                                                                                     newGen_np[true_tertiary_mask] != BOGUS_INT)))
            this_tagged_as_primary_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == 2)
            this_not_tagged_tertiary = np.count_nonzero(newGen_np[true_tertiary_mask] == BOGUS_INT)
            this_incorrect_parent_tertiary = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_tertiary_mask] == trueVisibleParentPFPIndex_np[true_tertiary_mask], \
                                                                                           np.logical_or(newGen_np[true_tertiary_mask] == 2, \
                                                                                                         newGen_np[true_tertiary_mask] == BOGUS_INT))))
            # Higher
            this_correct_parent_correct_tier_higher = 0
            this_correct_parent_wrong_tier_higher = np.count_nonzero(newParentPFPIndex_np[true_higher_mask] == trueVisibleParentPFPIndex_np[true_higher_mask])
            this_tagged_as_primary_higher = np.count_nonzero(newGen_np[true_higher_mask] == 2)
            this_not_tagged_higher = np.count_nonzero(newGen_np[true_higher_mask] == BOGUS_INT)
            this_incorrect_parent_higher = np.count_nonzero(np.logical_not(np.logical_or(newParentPFPIndex_np[true_higher_mask] == trueVisibleParentPFPIndex_np[true_higher_mask], \
                                                                                         np.logical_or(newGen_np[true_higher_mask] == 2, \
                                                                                                       newGen_np[true_higher_mask] == BOGUS_INT))))


            #############################################
            # Add metrics to global
            #############################################
            n_two_d += this_two_d
            n_true_primary += this_true_primary
            n_true_secondary += this_true_secondary
            n_true_tertiary += this_true_tertiary
            n_true_higher += this_true_higher

            n_correct_parent_correct_tier_primary += this_correct_parent_correct_tier_primary
            n_correct_parent_wrong_tier_primary += this_correct_parent_wrong_tier_primary
            n_tagged_as_primary_primary += this_tagged_as_primary_primary
            n_incorrect_parent_primary += this_incorrect_parent_primary
            n_not_tagged_primary += this_not_tagged_primary
            n_correct_parent_correct_tier_secondary += this_correct_parent_correct_tier_secondary
            n_correct_parent_wrong_tier_secondary += this_correct_parent_wrong_tier_secondary
            n_tagged_as_primary_secondary += this_tagged_as_primary_secondary
            n_incorrect_parent_secondary += this_incorrect_parent_secondary
            n_not_tagged_secondary += this_not_tagged_secondary
            n_correct_parent_correct_tier_tertiary += this_correct_parent_correct_tier_tertiary
            n_correct_parent_wrong_tier_tertiary += this_correct_parent_wrong_tier_tertiary
            n_tagged_as_primary_tertiary += this_tagged_as_primary_tertiary
            n_incorrect_parent_tertiary += this_incorrect_parent_tertiary
            n_not_tagged_tertiary += this_not_tagged_tertiary
            n_correct_parent_correct_tier_higher += this_correct_parent_correct_tier_higher
            n_correct_parent_wrong_tier_higher += this_correct_parent_wrong_tier_higher
            n_tagged_as_primary_higher += this_tagged_as_primary_higher
            n_incorrect_parent_higher += this_incorrect_parent_higher
            n_not_tagged_higher += this_not_tagged_higher

        #############################################
        # Print metrics
        #############################################   

        n_correct_parent_correct_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_correct_tier_primary) / float(n_true_primary), 2)
        n_correct_parent_wrong_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_wrong_tier_primary) / float(n_true_primary), 2)
        n_tagged_as_primary_primary_frac = round(0.0 if n_true_primary == 0 else float(n_tagged_as_primary_primary) / float(n_true_primary), 2)
        n_incorrect_parent_primary_frac = round(0.0 if n_true_primary == 0 else float(n_incorrect_parent_primary) / float(n_true_primary), 2)
        n_not_tagged_primary_frac = round(0.0 if n_true_primary == 0 else float(n_not_tagged_primary) / float(n_true_primary), 2)

        n_correct_parent_correct_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_correct_tier_secondary) / float(n_true_secondary), 2)
        n_correct_parent_wrong_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_wrong_tier_secondary) / float(n_true_secondary), 2)
        n_tagged_as_primary_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_tagged_as_primary_secondary) / float(n_true_secondary), 2)
        n_incorrect_parent_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_incorrect_parent_secondary) / float(n_true_secondary), 2)
        n_not_tagged_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_not_tagged_secondary) / float(n_true_secondary), 2)

        n_correct_parent_correct_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_correct_tier_tertiary) / float(n_true_tertiary), 2)
        n_correct_parent_wrong_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_wrong_tier_tertiary) / float(n_true_tertiary), 2)
        n_tagged_as_primary_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_tagged_as_primary_tertiary) / float(n_true_tertiary), 2)
        n_incorrect_parent_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_incorrect_parent_tertiary) / float(n_true_tertiary), 2)
        n_not_tagged_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_not_tagged_tertiary) / float(n_true_tertiary), 2)

        n_correct_parent_correct_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_correct_tier_higher) / float(n_true_higher), 2)
        n_correct_parent_wrong_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_wrong_tier_higher) / float(n_true_higher), 2)
        n_tagged_as_primary_higher_frac = round(0.0 if n_true_higher == 0 else float(n_tagged_as_primary_higher) / float(n_true_higher), 2)
        n_incorrect_parent_higher_frac = round(0.0 if n_true_higher == 0 else float(n_incorrect_parent_higher) / float(n_true_higher), 2)
        n_not_tagged_higher_frac = round(0.0 if n_true_higher == 0 else float(n_not_tagged_higher) / float(n_true_higher), 2)


        print('------------------------------------------------------------')
        print(('TRACK' if isTrack else 'SHOWER'))
        print('------------------------------------------------------------')
        print('NEW - True Gen   | Primary | Secondary | Tertiary | Higher |')
        print('------------------------------------------------------------')
        print('Correct parent CT |' + str(n_correct_parent_correct_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_correct_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_correct_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_correct_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_correct_tier_higher_frac)))) + \
                                '|')
        print('Correct parent WT |' + str(n_correct_parent_wrong_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_wrong_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_wrong_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_wrong_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_wrong_tier_higher_frac)))) + \
                                '|')
        print('False primary     |' + str(n_tagged_as_primary_primary_frac) + str(' '* (9 - len(str(n_tagged_as_primary_primary_frac)))) + \
                                '|' + str(n_tagged_as_primary_secondary_frac) + str(' '* (11 - len(str(n_tagged_as_primary_secondary_frac)))) + \
                                '|' + str(n_tagged_as_primary_tertiary_frac) + str(' '* (10 - len(str(n_tagged_as_primary_tertiary_frac)))) + \
                                '|' + str(n_tagged_as_primary_higher_frac) + str(' '* (8 - len(str(n_tagged_as_primary_higher_frac)))) + \
                                '|')
        print('Incorrect parent  |' + str(n_incorrect_parent_primary_frac) + str(' '* (9 - len(str(n_incorrect_parent_primary_frac)))) + \
                                '|' + str(n_incorrect_parent_secondary_frac) + str(' '* (11 - len(str(n_incorrect_parent_secondary_frac)))) + \
                                '|' + str(n_incorrect_parent_tertiary_frac) + str(' '* (10 - len(str(n_incorrect_parent_tertiary_frac)))) + \
                                '|' + str(n_incorrect_parent_higher_frac) + str(' '* (8 - len(str(n_incorrect_parent_higher_frac)))) + \
                                '|')
        print('Not tagged        |' + str(n_not_tagged_primary_frac) + str(' '* (9 - len(str(n_not_tagged_primary_frac)))) + \
                                '|' + str(n_not_tagged_secondary_frac) + str(' '* (11 - len(str(n_not_tagged_secondary_frac)))) + \
                                '|' + str(n_not_tagged_tertiary_frac) + str(' '* (10 - len(str(n_not_tagged_tertiary_frac)))) + \
                                '|' + str(n_not_tagged_higher_frac) + str(' '* (8 - len(str(n_not_tagged_higher_frac)))) + \
                                '|')
        print('------------------------------------------------------------')
        print('Total             |' + str(n_true_primary) + str(' '* (9 - len(str(n_true_primary)))) + \
                                '|' + str(n_true_secondary) + str(' '* (11 - len(str(n_true_secondary)))) + \
                                '|' + str(n_true_tertiary) + str(' '* (10 - len(str(n_true_tertiary)))) + \
                                '|' + str(n_true_higher) + str(' '* (8 - len(str(n_true_higher)))) + \
                                '|')
        print('n_two_d (not included in metrics):', n_two_d)
        print('------------------------------------------------------------')
        print('')
     
####################################################################################################################################### 
#######################################################################################################################################
    
def calculateMetrics_pandora(nEntries, particleMask_in, nSpacepoints_in, trackShowerScore_in, trueVisibleGeneration_in, trueVisibleParentSelf_in, pandoraParentSelf_in, pandoraGeneration_in) :       
    for isTrack in [True, False] : 

        ############################
        # Pandora metrics!
        #############################
        n_two_d = 0
        n_true_primary = 0
        n_true_secondary = 0
        n_true_tertiary = 0
        n_true_higher = 0

        n_correct_parent_correct_tier_primary_pandora = 0
        n_correct_parent_wrong_tier_primary_pandora = 0
        n_tagged_as_primary_primary_pandora = 0
        n_incorrect_parent_primary_pandora = 0
        n_not_tagged_primary_pandora = 0 

        n_correct_parent_correct_tier_secondary_pandora = 0
        n_correct_parent_wrong_tier_secondary_pandora = 0
        n_tagged_as_primary_secondary_pandora = 0
        n_incorrect_parent_secondary_pandora = 0
        n_not_tagged_secondary_pandora = 0 

        n_correct_parent_correct_tier_tertiary_pandora = 0
        n_correct_parent_wrong_tier_tertiary_pandora = 0
        n_tagged_as_primary_tertiary_pandora = 0
        n_incorrect_parent_tertiary_pandora = 0
        n_not_tagged_tertiary_pandora = 0 

        n_correct_parent_correct_tier_higher_pandora = 0
        n_correct_parent_wrong_tier_higher_pandora = 0
        n_tagged_as_primary_higher_pandora = 0
        n_incorrect_parent_higher_pandora = 0
        n_not_tagged_higher_pandora = 0 

        for iEvent in range(nEntries) : 

            # Particle mask
            particle_mask = np.array(particleMask_in[iEvent])
            # PFP info
            nSpacepoints_np = np.array(nSpacepoints_in[iEvent])[particle_mask]
            trackShowerScore_np = np.array(trackShowerScore_in[iEvent])[particle_mask]
            pandoraParentSelf_np = np.array(pandoraParentSelf_in[iEvent])[particle_mask]
            pandoraGeneration_np = np.array(pandoraGeneration_in[iEvent])[particle_mask]
            # Truth
            trueVisibleGeneration_np = np.array(trueVisibleGeneration_in[iEvent])[particle_mask]
            trueVisibleParentSelf_np = np.array(trueVisibleParentSelf_in[iEvent])[particle_mask]

            
            #########################
            # Get tier masks
            #########################
            ignore_mask = np.logical_or(nSpacepoints_np == 0, trueVisibleGeneration_np == -999)
            trackShower_mask = (trackShowerScore_np > 0.5) if isTrack else np.logical_not(trackShowerScore_np > 0.5)
            true_primary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 2)
            true_secondary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 3)
            true_tertiary_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), trueVisibleGeneration_np == 4)
            true_higher_mask = np.logical_and(np.logical_and(np.logical_not(ignore_mask), trackShower_mask), 
                                              np.logical_not(np.logical_or(true_primary_mask, np.logical_or(true_secondary_mask, true_tertiary_mask))))            

            #############################################
            # Get metrics for this event - debugging
            #############################################
            # Totals
            this_two_d = np.count_nonzero(np.logical_and(ignore_mask, trackShower_mask))
            this_true_primary = np.count_nonzero(true_primary_mask)
            this_true_secondary = np.count_nonzero(true_secondary_mask)
            this_true_tertiary = np.count_nonzero(true_tertiary_mask)
            this_true_higher = np.count_nonzero(true_higher_mask)

            # Primary
            this_correct_parent_correct_tier_primary_pandora = np.count_nonzero(pandoraGeneration_np[true_primary_mask] == 2)
            this_correct_parent_wrong_tier_primary_pandora = 0                                                                    
            this_tagged_as_primary_primary_pandora = 0
            this_not_tagged_primary_pandora = np.count_nonzero(pandoraGeneration_np[true_primary_mask] == BOGUS_INT)
            this_incorrect_parent_primary_pandora = np.count_nonzero(np.logical_and(pandoraGeneration_np[true_primary_mask] != 2, \
                                                                                pandoraGeneration_np[true_primary_mask] != BOGUS_INT)) 
            # Secondary
            this_correct_parent_correct_tier_secondary_pandora = np.count_nonzero(np.logical_and(pandoraParentSelf_np[true_secondary_mask] == trueVisibleParentSelf_np[true_secondary_mask], \
                                                                                                 pandoraGeneration_np[true_secondary_mask] == 3))
            this_correct_parent_wrong_tier_secondary_pandora = np.count_nonzero(np.logical_and(pandoraParentSelf_np[true_secondary_mask] == trueVisibleParentSelf_np[true_secondary_mask], \
                                                                                               np.logical_and(pandoraGeneration_np[true_secondary_mask] != 3, 
                                                                                                              pandoraGeneration_np[true_secondary_mask] != BOGUS_INT)))
            this_correct_parent_wrong_tier_secondary_pandora = np.count_nonzero(pandoraParentSelf_np[true_secondary_mask] == trueVisibleParentSelf_np[true_secondary_mask])
            this_tagged_as_primary_secondary_pandora = np.count_nonzero(pandoraGeneration_np[true_secondary_mask] == 2)
            this_not_tagged_secondary_pandora = np.count_nonzero(pandoraGeneration_np[true_secondary_mask] == BOGUS_INT)
            this_incorrect_parent_secondary_pandora = np.count_nonzero(np.logical_not(np.logical_or(pandoraParentSelf_np[true_secondary_mask] == trueVisibleParentSelf_np[true_secondary_mask], \
                                                                                                np.logical_or(pandoraGeneration_np[true_secondary_mask] == 2, \
                                                                                                pandoraGeneration_np[true_secondary_mask] == BOGUS_INT))))
            # Tertiary
            this_correct_parent_correct_tier_tertiary_pandora = np.count_nonzero(np.logical_and(pandoraParentSelf_np[true_tertiary_mask] == trueVisibleParentSelf_np[true_tertiary_mask], \
                                                                                                pandoraGeneration_np[true_tertiary_mask] == 4))
            this_correct_parent_wrong_tier_tertiary_pandora = np.count_nonzero(np.logical_and(pandoraParentSelf_np[true_tertiary_mask] == trueVisibleParentSelf_np[true_tertiary_mask], \
                                                                                              np.logical_and(pandoraGeneration_np[true_tertiary_mask] != 4, 
                                                                                                             pandoraGeneration_np[true_tertiary_mask] != BOGUS_INT)))
            this_tagged_as_primary_tertiary_pandora = np.count_nonzero(pandoraGeneration_np[true_tertiary_mask] == 2)
            this_not_tagged_tertiary_pandora = np.count_nonzero(pandoraGeneration_np[true_tertiary_mask] == BOGUS_INT)
            this_incorrect_parent_tertiary_pandora = np.count_nonzero(np.logical_not(np.logical_or(pandoraParentSelf_np[true_tertiary_mask] == trueVisibleParentSelf_np[true_tertiary_mask], \
                                                                                                np.logical_or(pandoraGeneration_np[true_tertiary_mask] == 2, \
                                                                                                pandoraGeneration_np[true_tertiary_mask] == BOGUS_INT))))
            # Higher
            this_correct_parent_correct_tier_higher_pandora = 0
            this_correct_parent_wrong_tier_higher_pandora = np.count_nonzero(pandoraParentSelf_np[true_higher_mask] == trueVisibleParentSelf_np[true_higher_mask])
            this_tagged_as_primary_higher_pandora = np.count_nonzero(pandoraGeneration_np[true_higher_mask] == 2)
            this_not_tagged_higher_pandora = np.count_nonzero(pandoraGeneration_np[true_higher_mask] == BOGUS_INT)
            this_incorrect_parent_higher_pandora = np.count_nonzero(np.logical_not(np.logical_or(pandoraParentSelf_np[true_higher_mask] == trueVisibleParentSelf_np[true_higher_mask], \
                                                                                                 np.logical_or(pandoraGeneration_np[true_higher_mask] == 2, \
                                                                                                               pandoraGeneration_np[true_higher_mask] == BOGUS_INT))))


            #############################################
            # Add metrics to global
            #############################################
            n_two_d += this_two_d
            n_true_primary += this_true_primary
            n_true_secondary += this_true_secondary
            n_true_tertiary += this_true_tertiary
            n_true_higher += this_true_higher

            n_correct_parent_correct_tier_primary_pandora += this_correct_parent_correct_tier_primary_pandora
            n_correct_parent_wrong_tier_primary_pandora += this_correct_parent_wrong_tier_primary_pandora
            n_tagged_as_primary_primary_pandora += this_tagged_as_primary_primary_pandora
            n_incorrect_parent_primary_pandora += this_incorrect_parent_primary_pandora
            n_not_tagged_primary_pandora += this_not_tagged_primary_pandora

            n_correct_parent_correct_tier_secondary_pandora += this_correct_parent_correct_tier_secondary_pandora
            n_correct_parent_wrong_tier_secondary_pandora += this_correct_parent_wrong_tier_secondary_pandora
            n_tagged_as_primary_secondary_pandora += this_tagged_as_primary_secondary_pandora
            n_incorrect_parent_secondary_pandora += this_incorrect_parent_secondary_pandora
            n_not_tagged_secondary_pandora += this_not_tagged_secondary_pandora

            n_correct_parent_correct_tier_tertiary_pandora += this_correct_parent_correct_tier_tertiary_pandora
            n_correct_parent_wrong_tier_tertiary_pandora += this_correct_parent_wrong_tier_tertiary_pandora
            n_tagged_as_primary_tertiary_pandora += this_tagged_as_primary_tertiary_pandora
            n_incorrect_parent_tertiary_pandora += this_incorrect_parent_tertiary_pandora
            n_not_tagged_tertiary_pandora += this_not_tagged_tertiary_pandora

            n_correct_parent_correct_tier_higher_pandora += this_correct_parent_correct_tier_higher_pandora
            n_correct_parent_wrong_tier_higher_pandora += this_correct_parent_wrong_tier_higher_pandora
            n_tagged_as_primary_higher_pandora += this_tagged_as_primary_higher_pandora
            n_incorrect_parent_higher_pandora += this_incorrect_parent_higher_pandora
            n_not_tagged_higher_pandora += this_not_tagged_higher_pandora

        #############################################
        # Print metrics
        #############################################   

        n_correct_parent_correct_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_correct_tier_primary_pandora) / float(n_true_primary), 2)
        n_correct_parent_wrong_tier_primary_frac = round(0.0 if n_true_primary == 0 else float(n_correct_parent_wrong_tier_primary_pandora) / float(n_true_primary), 2)
        n_tagged_as_primary_primary_frac = round(0.0 if n_true_primary == 0 else float(n_tagged_as_primary_primary_pandora) / float(n_true_primary), 2)
        n_incorrect_parent_primary_frac = round(0.0 if n_true_primary == 0 else float(n_incorrect_parent_primary_pandora) / float(n_true_primary), 2)
        n_not_tagged_primary_frac = round(0.0 if n_true_primary == 0 else float(n_not_tagged_primary_pandora) / float(n_true_primary), 2)

        n_correct_parent_correct_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_correct_tier_secondary_pandora) / float(n_true_secondary), 2)
        n_correct_parent_wrong_tier_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_correct_parent_wrong_tier_secondary_pandora) / float(n_true_secondary), 2)
        n_tagged_as_primary_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_tagged_as_primary_secondary_pandora) / float(n_true_secondary), 2)
        n_incorrect_parent_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_incorrect_parent_secondary_pandora) / float(n_true_secondary), 2)
        n_not_tagged_secondary_frac = round(0.0 if n_true_secondary == 0 else float(n_not_tagged_secondary_pandora) / float(n_true_secondary), 2)

        n_correct_parent_correct_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_correct_tier_tertiary_pandora) / float(n_true_tertiary), 2)
        n_correct_parent_wrong_tier_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_correct_parent_wrong_tier_tertiary_pandora) / float(n_true_tertiary), 2)
        n_tagged_as_primary_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_tagged_as_primary_tertiary_pandora) / float(n_true_tertiary), 2)
        n_incorrect_parent_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_incorrect_parent_tertiary_pandora) / float(n_true_tertiary), 2)
        n_not_tagged_tertiary_frac = round(0.0 if n_true_tertiary == 0 else float(n_not_tagged_tertiary_pandora) / float(n_true_tertiary), 2)    

        n_correct_parent_correct_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_correct_tier_higher_pandora) / float(n_true_higher), 2)
        n_correct_parent_wrong_tier_higher_frac = round(0.0 if n_true_higher == 0 else float(n_correct_parent_wrong_tier_higher_pandora) / float(n_true_higher), 2)
        n_tagged_as_primary_higher_frac = round(0.0 if n_true_higher == 0 else float(n_tagged_as_primary_higher_pandora) / float(n_true_higher), 2)
        n_incorrect_parent_higher_frac = round(0.0 if n_true_higher == 0 else float(n_incorrect_parent_higher_pandora) / float(n_true_higher), 2)
        n_not_tagged_higher_frac = round(0.0 if n_true_higher == 0 else float(n_not_tagged_higher_pandora) / float(n_true_higher), 2)

        print('-------------------------------------------------------------')
        print(('PANDORA' if isTrack else 'SHOWER'))
        print('-------------------------------------------------------------')
        print('NEW - True Gen    | Primary | Secondary | Tertiary | Higher |')
        print('-------------------------------------------------------------')
        print('Correct parent CT |' + str(n_correct_parent_correct_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_correct_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_correct_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_correct_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_correct_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_correct_tier_higher_frac)))) + \
                                '|')
        print('Correct parent WT |' + str(n_correct_parent_wrong_tier_primary_frac) + str(' '* (9 - len(str(n_correct_parent_wrong_tier_primary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_secondary_frac) + str(' '* (11 - len(str(n_correct_parent_wrong_tier_secondary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_tertiary_frac) + str(' '* (10 - len(str(n_correct_parent_wrong_tier_tertiary_frac)))) + \
                                '|' + str(n_correct_parent_wrong_tier_higher_frac) + str(' '* (8 - len(str(n_correct_parent_wrong_tier_higher_frac)))) + \
                                '|')
        print('False primary     |' + str(n_tagged_as_primary_primary_frac) + str(' '* (9 - len(str(n_tagged_as_primary_primary_frac)))) + \
                                '|' + str(n_tagged_as_primary_secondary_frac) + str(' '* (11 - len(str(n_tagged_as_primary_secondary_frac)))) + \
                                '|' + str(n_tagged_as_primary_tertiary_frac) + str(' '* (10 - len(str(n_tagged_as_primary_tertiary_frac)))) + \
                                '|' + str(n_tagged_as_primary_higher_frac) + str(' '* (8 - len(str(n_tagged_as_primary_higher_frac)))) + \
                                '|')
        print('Incorrect parent  |' + str(n_incorrect_parent_primary_frac) + str(' '* (9 - len(str(n_incorrect_parent_primary_frac)))) + \
                                '|' + str(n_incorrect_parent_secondary_frac) + str(' '* (11 - len(str(n_incorrect_parent_secondary_frac)))) + \
                                '|' + str(n_incorrect_parent_tertiary_frac) + str(' '* (10 - len(str(n_incorrect_parent_tertiary_frac)))) + \
                                '|' + str(n_incorrect_parent_higher_frac) + str(' '* (8 - len(str(n_incorrect_parent_higher_frac)))) + \
                                '|')
        print('Not tagged        |' + str(n_not_tagged_primary_frac) + str(' '* (9 - len(str(n_not_tagged_primary_frac)))) + \
                                '|' + str(n_not_tagged_secondary_frac) + str(' '* (11 - len(str(n_not_tagged_secondary_frac)))) + \
                                '|' + str(n_not_tagged_tertiary_frac) + str(' '* (10 - len(str(n_not_tagged_tertiary_frac)))) + \
                                '|' + str(n_not_tagged_higher_frac) + str(' '* (8 - len(str(n_not_tagged_higher_frac)))) + \
                                '|')
        print('-------------------------------------------------------------')
        print('Total             |' + str(n_true_primary) + str(' '* (9 - len(str(n_true_primary)))) + \
                                '|' + str(n_true_secondary) + str(' '* (11 - len(str(n_true_secondary)))) + \
                                '|' + str(n_true_tertiary) + str(' '* (10 - len(str(n_true_tertiary)))) + \
                                '|' + str(n_true_higher) + str(' '* (8 - len(str(n_true_higher)))) + \
                                '|')
        print('n_two_d (not included in metrics):', n_two_d)
        print('-------------------------------------------------------------')
        print('')

####################################################################################################################################### 
#######################################################################################################################################
def InvTrainingCuts(targetMask_in, trueVisibleParentPFPIndex_in, newParentPFPIndex_in, pfpIndex_in, \
                    higherLinkMask_in, parentPFPIndex_in, childPFPIndex_in, trainingCutL_in, trainingCutT_in) :

    trainingCutL_correct = []
    trainingCutT_correct = []
    
    trainingCutL_incorrect = []
    trainingCutT_incorrect = []
    
    for iEvent in range(targetMask_in.shape[0]) :
        
        # If no target particles
        if (np.count_nonzero(targetMask_in[iEvent]) == 0) :
            continue
            
        # Node
        trueVisibleParentPFPIndex = trueVisibleParentPFPIndex_in[iEvent][targetMask_in[iEvent]]
        newParentPFPIndex = newParentPFPIndex_in[iEvent][targetMask_in[iEvent]]
        pfpIndex = pfpIndex_in[iEvent][targetMask_in[iEvent]]

        # Edges
        parentPFPIndex = parentPFPIndex_in[iEvent][higherLinkMask_in[iEvent]]
        childPFPIndex = childPFPIndex_in[iEvent][higherLinkMask_in[iEvent]]
        trainingCutL = trainingCutL_in[iEvent][higherLinkMask_in[iEvent]]
        trainingCutT = trainingCutT_in[iEvent][higherLinkMask_in[iEvent]]
        
        for i in range(pfpIndex.shape[0]) :
            
            invLinkIndex = np.where(np.logical_and(childPFPIndex == pfpIndex[i], parentPFPIndex == trueVisibleParentPFPIndex[i]))[0]
            
            if (len(invLinkIndex)) == 0 :
                continue
                
            if (newParentPFPIndex[i] == trueVisibleParentPFPIndex[i]) :
                trainingCutL_correct.append(trainingCutL[invLinkIndex[0]])
                trainingCutT_correct.append(trainingCutT[invLinkIndex[0]])
            else :
                trainingCutL_incorrect.append(trainingCutL[invLinkIndex[0]])
                trainingCutT_incorrect.append(trainingCutT[invLinkIndex[0]])
                
#     this_correct_parent = (newParentPFPIndex == trueVisibleParentPFPIndex)
#     this_incorrect_parent = np.logical_not(this_correct_parent)
    
#     trainingCutL_correct = trainingCutL[this_correct_parent]
#     trainingCutL_incorrect = trainingCutL[this_incorrect_parent]

#     trainingCutT_correct = trainingCutT[this_correct_parent]
#     trainingCutT_incorrect = trainingCutT[this_incorrect_parent]

    lowT = 0.0
    highT = 100.0
    
    lowL = -200.0
    highL = 200.0


    plt.clf()
    plt.hist(trainingCutT_correct, range=[lowT, highT], bins=50, color='green', label='correct', histtype='step')
    plt.hist(trainingCutT_incorrect, range=[lowT, highT],bins=50, color='red', label='incorrect', histtype='step')
    plt.xlabel('trainingCutT')
    plt.ylabel('arbitary units')
    plt.legend()
    plt.show()
    
    plt.hist(trainingCutL_correct, range=[lowL, highL], bins=50, color='green', label='correct', histtype='step')
    plt.hist(trainingCutL_incorrect, range=[lowL, highL],bins=50, color='red', label='incorrect', histtype='step')
    plt.xlabel('trainingCutL')
    plt.ylabel('arbitary units')
    plt.legend()
    plt.show()

    plt.hist2d(trainingCutL_correct, trainingCutT_correct, bins=100, range=[[lowL, highL], [lowT, highT]])
    plt.title('correct')
    plt.xlabel('trainingCutL')
    plt.ylabel('trainingCutT')
    plt.legend()
    plt.show()
    
    plt.hist2d(trainingCutL_incorrect, trainingCutT_incorrect, bins=100, range=[[lowL, highL], [lowT, highT]])
    plt.title('incorrect')
    plt.xlabel('trainingCutL')
    plt.ylabel('trainingCutT')
    plt.legend()
    plt.show()
    #plt.hist2d(trainingCutL_incorrect, trainingCutT_incorrect, bins=100, range=[[-500, 500], [0, 500]])
        