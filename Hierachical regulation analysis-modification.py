import os
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from scipy.stats.distributions import chi2
from scipy.stats import pearsonr
pd.set_option('display.width',None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def r_rdataframe(reaction_ID,step,Reactions):
    '''
    :param reaction_ID: Reaction_ID in GEMs of S.cerevisiae (GEMs Yeast 7.6)
    :param step:  the reaction being processed
    :param Reactions: A dataframe including all the reactions
    :return:
    A reactant dataframe. For example:
    Reaction_ID   Reactants_ID    Measured_Value    Km                   Km_value
    r_0466        s_0568           True             0.127921035814782     True
    r_0466        s_1207           True             0.0238522852342474    True
    r_0466        s_0803           False            NA                    False
    r_0466        s_0335           False            NA                    False

    True means data are measured (for reactants) or cited (for Km), while false is inverse.
    Reactants whose "Measured_Value" and km_value are true are taken into consideration.
    '''

    reactants_series = Reactions.iloc[step, 1]                            # acquire reactants
    Km_series=str(Reactions.iloc[step,4])                                 # acquire Km
    space_number = len(list(a for a in reactants_series if a == " "))     # acquire the space numbers in reactants
    reactants_dataframe = pd.DataFrame(np.random.rand((space_number + 1) * 5).reshape(space_number + 1, 5),
                                    columns=['Reaction_ID', 'Reactants_ID', 'Measured_Value','Km','Km_value'])

    for b in range(0, space_number + 1):
        reactants_involved = reactants_series.split(' ', space_number)[b]      # split all the reactants into a single reactant according to the sapce numbers
        reactants_dataframe.iloc[b, 1] = reactants_involved
        reactants_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == reactants_involved)]    # match the reactant to another dataframe ("Metabolites") to acquire the measurements of reactant
        Km_involved=Km_series.split(' ',space_number)[b]          # split all the Km according to the space numbers
        reactants_dataframe.iloc[b,3]=Km_involved

        if pd.isnull(reactants_screened_name['MS1'].tolist()) or np.any(reactants_screened_name == 0):   # the reactant measurements which constains zero or "Nan" will be considered false
            reactants_measured_value = "False"
            reactants_dataframe.iloc[b, 2] = reactants_measured_value
        else:
            reactants_measured_value = "True"
            reactants_dataframe.iloc[b, 2] = reactants_measured_value


        if Km_involved=='nan' or Km_involved=='0' or Km_involved=='NA':         # the Km values which constains zero or "NA" will be considered false
            Km_value="False"
            reactants_dataframe.iloc[b, 4] = Km_value
        else:
            Km_value="True"
            reactants_dataframe.iloc[b, 4] = Km_value
    reactants_dataframe['Reaction_ID'] = reaction_ID
    return reactants_dataframe

def r_edataframe(step,Reactions,reactants_dataframe,reaction_ID):
    '''

    :param step:      the reaction being processed
    :param Reactions:  A dataframe including all the reactions
    :param reactants_dataframe: a reactant dataframe including reaction_ID, reactants, and Km
    :param reaction_ID:    Reaction_ID in GEMs of S.cerevisiae (GEMs Yeast 7.6)
    :return:
    An enzyme dataframe.For example:
    Reaction_ID   Enzyme_ID    Measured_Value
    r_0466        YNL241C            True
    r_0466        YGR248W            False
    r_0466        YHR163W            True

    True means data are measured, while false is inverse.
    Enzymes whose "Measured_Value" is true are taken into consideration.
    '''

    if ("True" in list(reactants_dataframe['Measured_Value']) and "True" in list(reactants_dataframe['Km_value'])):
        # Proceed to the next step when ensuring that any reactant in a reaction can be measured.

        enzyme_ID = Reactions.iloc[step, 2]                   # acquire enzymes ID
        space_number_2 = len(list(a for a in enzyme_ID if a == " "))       # acquire the space numbers in enzymes ID
        enzyme_dataframe = pd.DataFrame(np.random.rand((space_number_2 + 1) * 3).reshape(space_number_2 + 1, 3),
                                    columns=['Reaction_ID', 'Enzyme_ID', 'Measured_Value'])

        enzyme_primary_matrix = np.array([0])
        enzyme_primary_measured_matrix = np.array([0])

        for c in range(0, space_number_2 + 1):
            enzyme_involved = enzyme_ID.split(' ', space_number_2)[c]      # split all the enzymes ID into a single enzyme ID according to the sapce numbers
            enzyme_second_matrix = np.array([enzyme_involved])
            enzyme_primary_matrix = np.append(enzyme_primary_matrix, enzyme_second_matrix, axis=0)
            enzyme_second_measured_matrix = Enzymes.loc[(Enzymes['Gene'] == enzyme_involved)]   # match the enzyme ID to another dataframe ("Enzymes") to acquire the measurements of enzyme.

            if pd.isnull(enzyme_second_measured_matrix['MS1'].tolist()) or np.any(
                    enzyme_second_measured_matrix == 0) or enzyme_second_measured_matrix.size == 0:    # the enzyme measurements which constains zero or "Nan" will be considered false.
                enzyme_measured_value = "False"
                enzyme_primary_measured_matrix = np.append(enzyme_primary_measured_matrix,
                                                           np.array([enzyme_measured_value]), axis=0)
            else:
                enzyme_measured_value = "True"
                enzyme_primary_measured_matrix = np.append(enzyme_primary_measured_matrix,
                                                           np.array([enzyme_measured_value]), axis=0)

        enzyme_third_matrix = np.delete(enzyme_primary_matrix, 0, 0)
        enzyme_third_measured_matrix = np.delete(enzyme_primary_measured_matrix, 0, 0)
        enzyme_dataframe['Enzyme_ID'] = enzyme_third_matrix
        reaction_second_matrix = np.full(shape=(space_number_2 + 1), fill_value=reaction_ID)
        enzyme_dataframe['Reaction_ID'] = reaction_second_matrix
        enzyme_dataframe['Measured_Value'] = enzyme_third_measured_matrix

    else:
        # the cycle ends and next reaction will start when no reactants involved in the reaction are measured.
        print("metabolites or Km haven't been measured")
        enzyme_dataframe = pd.DataFrame(columns=['Reaction_ID', 'Enzyme_ID', 'Isoenzyme', 'Measured_Value'])   # return an empty dataframe
    return enzyme_dataframe



def a_dataframe(reaction_ID,proposed_enzyme_ID,Enzyme_Transform):
    """

    :param reaction_ID: Reaction_ID in GEMs of S.cerevisiae (GEMs Yeast 7.6)
    :param proposed_enzyme_ID: a list including all the enzymes whose values are measured
    :param Enzyme_Transform: A dataframe including all the enzymes and their inhibitors and activators
    :return:
    an inhibitors_dataframe. For example:
        Reaction_ID   Inhibitors_ID    Inhibitors_Value
    r_0466             s_1360            True
    r_0466             s_0423            True
    r_0466             s_1212            True

    an activators_dataframe. For example:
    Reaction_ID   Activators_ID    Activators_Value
    r_0466        False            False

    True means data are measured, while false is inverse.
    Inhibitors or activators whose "Measured_Value" is true are taken into consideration.

    The flow chart to acquire the "inhibitor dataframe" and "activator dataframe" is as follows:

    first: the inhibitors and activators of each reaction are collected according to the enzymes involved in the reaction (allosteric regulators of enzymes were cited from the BRENDA database and placed in the "Enzyme_Transform" dataframe ).
    second: all the allosteric regulators in each reaction are split into a single inhibitor or activator to determine whether the single regulator is measured.

    Annotation: "reaction inhibitors or reaction activators": inhibitors or activators for a reaction; "enzyme inhibitors or enzyme activators": inhibitors or activators for an enzyme.

    "Reaction inhibitors or reaction activators" are a set in which "enzyme inhibitors or enzyme activators" are assembled together due to a reaction having different enzymes. For example:
    For the reaction of r_1050, which has two enzymes, "YBR117C" and "YPR074C", the allosteric regulators are cited from the Brenda database according to the enzyme ID.
    Thus, "YBR117C" has three possible regulators, "s_1408, s_2966", and "s_1360";  "YPR074C" also has three possible regulators, "s_1408, s_2966", and "s_1360". All of these regulators are called "enzyme inhibitors or enzyme activators";
    Then, for a reaction, all of these inhibitors or reaction activators should be collected, and duplicates are removed. Therefore, "s_1408 s_2966 s_1360" are called "reaction inhibitors or reaction activators".
    """

    inhibitors_dataframe = pd.DataFrame(
        columns=['Reaction_ID', 'Inhibitors_ID', 'Inhibitors_Value'])
    activitors_dataframe = pd.DataFrame(
        columns=['Reaction_ID',  'Activators_ID',  'Activators_Value'])
    inh_enzyme_first_list=list()        # construct an empty list to contain all the "reaction inhibitors"
    act_enzyme_first_list=list()        # construct an empty list to contain all the "reaction activators"

    for d in proposed_enzyme_ID:      # collect all the "reaction inhibitors and reaction activators" according to the enzyme involved in the reaction
        a_enzyme_involved=Enzyme_Transform.loc[(Enzyme_Transform['Enzyme_ID']==d)]  #match the enzyme ID to a dataframe ("Enzyme_Transform") to acquire the "enzyme inhibitors or enzyme activators"
        inh_involved=list(a_enzyme_involved.iloc[:,1])
        act_involved=list(a_enzyme_involved.iloc[:,2])

        if pd.isnull(inh_involved):   # no "enzyme inhibitors"
            inh_enzyme_first_list = inh_enzyme_first_list+list()    # add "empty" to the "reaction inhibitors"
        else:
            inh_enzyme_first_list = inh_enzyme_first_list+inh_involved     #add the "enzyme inhibitors" to "reaction inhibitors"

        if pd.isnull(act_involved):          # no "enzyme activators"
            act_enzyme_first_list = act_enzyme_first_list+list()       # add "empty" to the "reaction activators"
            #delete the activator
        else:
            act_enzyme_first_list = act_enzyme_first_list+act_involved    # add the "enzyme inhibitors" to "reaction activators"

    if len(inh_enzyme_first_list)==0:   # no inhibitors for the "reaction inhibitors"
        inhibitors_dataframe['Inhibitors_ID']= np.array(['False'])
        inhibitors_dataframe['Inhibitors_Value'] = np.array(['False'])
        inhibitors_dataframe['Reaction_ID'] = reaction_ID

    else:     # "reaction inhibitors" has inhibitors
        big_inh_enzyme_first_matrix = np.array([0])    # construct an empty array to contain all single inbihitors from "reaction inhibitors"
        big_inh_enzyme_value_first_matrix = np.array([0])    # construct an empty array to contian the results that the inhibitor from "reaction inhibitors" is whether measured

        for inhibitor_reg in inh_enzyme_first_list:
            space_number_3 = len(list(a for a in inhibitor_reg if a == " "))  # acquire the space numbers in all inhibitors
            inh_enzyme_first_matrix = np.array([0])
            inh_enzyme_value_first_matrix = np.array([0])

            for e in range(0, space_number_3 + 1):
                a_inh_involved = inhibitor_reg.split(' ', space_number_3)[e]  # split all the inhibitors into a single inhibitor according to the sapce numbers
                inh_enzyme_first_matrix = np.append(inh_enzyme_first_matrix, np.array([a_inh_involved]), axis=0)   # add a single inhibitor into an array

                inh_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == a_inh_involved)]   # match single inhibitor to another dataframe ("Metabolites") to acquire the mearsurements of the inhibitor.

                if pd.isnull(inh_screened_name['MS1'].tolist()) or np.any(
                        inh_screened_name == 0):       # the inhititor measurements constain zero or "Nan"
                    inh_measured_value = "False"
                    inh_enzyme_value_first_matrix = np.append(inh_enzyme_value_first_matrix, np.array([inh_measured_value]),
                                                              axis=0)

                else:      # the inhititor measurements do not constainss zero or "Nan"
                    inh_measured_value = "True"
                    inh_enzyme_value_first_matrix = np.append(inh_enzyme_value_first_matrix, np.array([inh_measured_value]),
                                                              axis=0)

            inh_enzyme_second_matrix = np.delete(inh_enzyme_first_matrix, 0, 0)
            inh_enzyme_value_second_matrix = np.delete(inh_enzyme_value_first_matrix, 0, 0)
            big_inh_enzyme_first_matrix = np.append(big_inh_enzyme_first_matrix, inh_enzyme_second_matrix, axis=0)
            big_inh_enzyme_value_first_matrix = np.append(big_inh_enzyme_value_first_matrix,
                                                              inh_enzyme_value_second_matrix, axis=0)

        big_inh_enzyme_second_matrix = np.delete(big_inh_enzyme_first_matrix, 0, 0)
        big_inh_enzyme_value_second_matrix = np.delete(big_inh_enzyme_value_first_matrix, 0, 0)
        inhibitors_dataframe['Inhibitors_ID'] = big_inh_enzyme_second_matrix
        inhibitors_dataframe['Inhibitors_Value'] =big_inh_enzyme_value_second_matrix
        inhibitors_dataframe['Reaction_ID']=reaction_ID

    if len(act_enzyme_first_list)==0:    # no activators for the "reaction activators"
        activitors_dataframe['Activators_ID']= np.array(['False'])
        activitors_dataframe['Activators_Value'] = np.array(['False'])
        activitors_dataframe['Reaction_ID'] = reaction_ID
    else:   # "reaction activators" has activators
        big_act_enzyme_first_matrix = np.array([0])    # construct an empty array to contain all single activators from "reaction activators"
        big_act_enzyme_value_first_matrix = np.array([0])   # construct an empty array to contian the results that the actiavtors from "reaction activators" is whether measured

        for activators_reg in act_enzyme_first_list:
            space_number_4 = len(list(a for a in activators_reg if a == " "))    # acquire the space numbers in all activators
            act_enzyme_first_matrix = np.array([0])
            act_enzyme_value_first_matrix = np.array([0])

            for f in range(0, space_number_4 + 1):
                a_act_involved = activators_reg.split(' ', space_number_4)[f]   #split all the activators into a single activator according to the sapce numbers
                act_enzyme_first_matrix = np.append(act_enzyme_first_matrix, np.array([a_act_involved]), axis=0)  # add a single activator into an array
                act_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == a_act_involved)]  # match single activator to another dataframe ("Metabolites") to acquire the mearsurements of the activator.

                if pd.isnull(act_screened_name['MS1'].tolist()) or np.any(
                        act_screened_name == 0):   # the activator measurements constain zero or "Nan"
                    act_measured_value = "False"
                    act_enzyme_value_first_matrix = np.append(act_enzyme_value_first_matrix, np.array([act_measured_value]),
                                                              axis=0)

                else:    # the activator measurements do not constainss zero or "Nan"
                    act_measured_value = "True"
                    act_enzyme_value_first_matrix = np.append(act_enzyme_value_first_matrix, np.array([act_measured_value]),
                                                              axis=0)

            act_enzyme_second_matrix = np.delete(act_enzyme_first_matrix, 0, 0)
            act_enzyme_value_second_matrix = np.delete(act_enzyme_value_first_matrix, 0, 0)
            big_act_enzyme_first_matrix = np.append(big_act_enzyme_first_matrix, act_enzyme_second_matrix, axis=0)
            big_act_enzyme_value_first_matrix = np.append(big_act_enzyme_value_first_matrix,
                                                              act_enzyme_value_second_matrix, axis=0)

        big_act_enzyme_second_matrix = np.delete(big_act_enzyme_first_matrix, 0, 0)
        big_act_enzyme_value_second_matrix = np.delete(big_act_enzyme_value_first_matrix, 0, 0)
        activitors_dataframe['Activators_ID'] = big_act_enzyme_second_matrix
        activitors_dataframe['Activators_Value'] = big_act_enzyme_value_second_matrix
        activitors_dataframe['Reaction_ID'] = reaction_ID
    return inhibitors_dataframe, activitors_dataframe



def r_evalue(proposed_enzyme_ID,Enzymes):

    '''
    all the enzyme values are summed together to acquire the enzyme abundance

    :param proposed_enzyme_ID:a list including all the enzymes whose values are measured
    :param Enzymes: A dataframe including all the enzyme values under different specific growth conditions
    :return: An enzyme abundance array

    '''

    enzyme_first_value_matrix = np.mat(np.zeros(10))     # construct an array to contain all the enzyme measurements
    for g in proposed_enzyme_ID:
        enzyme_second_value_matrix = Enzymes.loc[(Enzymes['Gene'] == g)]   # match the enzyme ID to another dataframe ("Enzymes") to acquire the measurements of enzyme.
        enzyme_fifth_value_matrix = np.mat(enzyme_second_value_matrix.iloc[:, 0:10])
        enzyme_first_value_matrix = np.append(enzyme_first_value_matrix,
                                                    enzyme_fifth_value_matrix, axis=0)

    enzyme_third_value_matrix = np.delete(enzyme_first_value_matrix, 0, 0)
    enzyme_forth_value_matrix = np.delete(enzyme_third_value_matrix, 0, 1)
    enzyme_calculation = (np.array(np.transpose(enzyme_forth_value_matrix.sum(axis=0)))).astype(float)   # sum all the enzyme values involved in a reaction
    return enzyme_calculation

def met_judge(reactants_dataframe,proposed_inhibitors_ID,proposed_activators_ID):
    '''
    the reactants that are allosteric regulators are extracted from the primary reactants list into another list and removed from the primary allsoteric list

    :param reactants_dataframe: reactant dataframe including reaction_ID, reactants, and Km

    :param proposed_inhibitors_ID:  a list including all the inhibitors whose value are true in inhibitors_dataframe

    :param proposed_activators_ID: a list including all the activators whose value are true in inhibitors_dataframe

    :return: six lists (the first list is a list that contains all the measured reactants (primary reactants list); the second list is that the reactants are neither inhibitors nor activators;
    the third list is that the reactants are also inhibitors; the fourth list is that the reactants are also activators; the fifth list is an inhibitor list whose elements are removed compared to the primary reactants list;
    The sixth list is an inhibitor list whose elements are removed compared to the primary reactant list). For example:

    for a reactants_dataframe like this:
    Reaction_ID   Reactants_ID    Measured_Value    Km                  Km_value
    r_0000        a                 True            1                    True
    r_0000        b                 True            1                    True
    r_0000        c                 True            1                    True
    r_0000        d                 True            1                    True

    suppose: proposed_inhibitors_ID I=[a,e,f]; proposed_activators_ID A=[c,g,h];

    Then, we will obtain six lists: reactants_list= [a,b,c,d]; reactants_noallostery_list=[a,d]; reactants_inhibitors_list= [a]; reactants_activators_list=[c]; proposed_inhibitors_ID=[e,f]; proposed_activators_ID=[g,h];

    '''
    proposed_reactants_ID1 = list(reactants_dataframe['Reactants_ID'][reactants_dataframe['Measured_Value'] == 'True'])    # select measured reactants
    proposed_reactants_ID2 = list(reactants_dataframe['Reactants_ID'][reactants_dataframe['Km_value'] == 'True'])   # select Km whose value is measured

    reactants_all_list=list()   # construct an empty list to contain the reactants whose value and Km are measured

    for j in  proposed_reactants_ID1:
        if j in proposed_reactants_ID2:
            reactants_all_list=reactants_all_list+j.split()     # add the reactant whose value and Km are also measured into a list


    reactants_inhibitors_list=list()       # construct an empty list to contain the reactants whose are also inhibitors
    reactants_noinhibitors_list=list()     # construct an empty list to contain the reactants whose are not inhibitors

    for k in reactants_all_list:
        if k in proposed_inhibitors_ID:
            reactants_inhibitors_list=reactants_inhibitors_list+k.split()   # add the reactant whose are also inhibitors into a list
            proposed_inhibitors_ID.remove(k)      # remove the reactants in inhibitors list

        else:
            reactants_noinhibitors_list=reactants_noinhibitors_list+k.split()      # add the reactant whose are not inhibitors into a list

    reactants_activators_list = list()     # construct an empty list to contain the reactants whose are also activators
    reactants_noactivators_list = list()   # construct an empty list to contain the reactants whose are not activators

    for l in reactants_all_list:
        if l in proposed_activators_ID:
            reactants_activators_list = reactants_activators_list + l.split()   # add the reactant whose are also activators into a list
            proposed_activators_ID.remove(l)      # remove the reactants in activators list
        else:
            reactants_noactivators_list = reactants_noactivators_list + l.split()      # add the reactant whose are not activators into a list


    reactants_noallostery_list=list(set(reactants_noinhibitors_list).intersection(set(reactants_noactivators_list)))  # sum all the reactant whose are not neither inhibitors nor activators


    return (reactants_all_list,reactants_noallostery_list,reactants_inhibitors_list,reactants_activators_list, proposed_inhibitors_ID,proposed_activators_ID)


def r_mvalue(metabolite_list,Metabolites):
    '''

    :param metabolite_list: a list including all the metabolites whose values are true.
    :param Metabolites:    a dataframe including all the metabolite concentrations.
    :return: a logarithmic array that is the ratio between metabolite concentrations and Km
    '''
    if len(metabolite_list)==0:   # metabolite list is empty
        reactants_eighth_value_matrix=(np.array(np.zeros(9))).astype(float)

    else:
        reactants_first_value_matrix = np.mat(np.ones(9))   # construct an array to contain all the ratio between metabolite concentrations and Km

        for h in metabolite_list:
            reactants_second_value_matrix = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == h)]     # match the metabolite ID to another dataframe ("Metabolites") to acquire the measurements of metabolite
            reactants_third_value_matrix = np.delete(np.mat(reactants_second_value_matrix.iloc[:, 0:10]), 0,
                                                     1)
            Km = float((np.mat(reactants_dataframe['Km'][reactants_dataframe['Reactants_ID'] == h]))[0, 0])  # match the metabolite ID to another dataframe ("reactants_dataframe") to acquire the Km of metabolite

            reactants_forth_value_matrix = reactants_third_value_matrix / Km          # calcualte the ratio between metabolite concentrations and Km
            reactants_first_value_matrix = np.append(reactants_first_value_matrix, reactants_forth_value_matrix,
                                                     axis=0)

        reactants_fifth_value_matrix = np.delete(reactants_first_value_matrix, 0, 0)
        reactants_sixth_value_matrix = np.log(np.array(reactants_fifth_value_matrix, dtype='float'))   # calculate the logarithm
        reactants_eighth_value_matrix = (np.array(np.transpose(reactants_sixth_value_matrix))).astype(float)
    return(reactants_eighth_value_matrix)


def r_avalue(allosteric_value_list,Metabolites):
    '''

    :param allosteric_value_list: a list including all the allosteric modulators whose value is true
    :param Metabolites: a dataframe including all the metabolite concentrations.
    :return: a logarithmic array of allosteric regulator concentrations
    '''
    allosteric_first_value_matrix = np.mat(np.ones(9))       # construct a matrix to contain allosteric regulator concentrations
    for i in allosteric_value_list:
        allosteric_second_value_matrix = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == i)]  # match the allosteric regulators ID to a dataframe ("Metabolites") to acquire the measurements of the allosteric regulators

        allosteric_third_value_matrix = np.delete(np.mat(allosteric_second_value_matrix.iloc[:, 0:10]), 0,
                                                  1)
        allosteric_first_value_matrix = np.append(allosteric_first_value_matrix, allosteric_third_value_matrix,
                                                  axis=0)

    allosteric_fifth_value_matrix = np.delete(allosteric_first_value_matrix, 0, 0)
    allosteric_sixth_value_matrix = np.log(np.array(allosteric_fifth_value_matrix, dtype='float'))
    allosteric_eighth_value_matrix = (np.array(np.transpose(allosteric_sixth_value_matrix))).astype(float)
    return allosteric_eighth_value_matrix



def r_savalue(allosteric_regulators,Metabolites):
    '''

    :param allosteric_regulators:  a allosteric regulator;
    :param Metabolites:  a dataframe including all the metabolite concentrations;
    :return: a logarithmic matrix of allosteric regulator concentrations;
    '''
    allosteric_regulators_matrix1=Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == allosteric_regulators)]  # match the a allosteric regulator ID to a dataframe ("Metabolites") to acquire the measurements of the allosteric regulator
    allosteric_regulators_matrix2 = np.delete(np.mat(allosteric_regulators_matrix1.iloc[:, 0:10]), 0,
                                              1)
    allosteric_regulators_matrix3 = np.log(np.array(allosteric_regulators_matrix2, dtype='float'))
    allosteric_regulators_matrix6=(np.array(np.transpose(allosteric_regulators_matrix3))).astype(float)
    return allosteric_regulators_matrix6




def r_fvalue(reaction_ID,Fluxes,Fluxes_boundary):
    '''
    :param reaction_ID: Reaction_ID in GEMs of S.cerevisiae (GEMs Yeast 7.6)
    :param Fluxes: a dataframe including all the flux values;
    :param Fluxes_boundary: a dataframe including all the lower and upper boundaries of flux values;
    :return: a flux array; an array containing the lower boundary of flux values; an array including the upper boundary of flux values
    '''

    Flux_first_value_matrix = Fluxes.loc[(Fluxes['Model_Reaction_ID'] == reaction_ID)]   # match the reaction_ID to another dataframe ("Fluxes") to acquire the measurements of the flux
    Flux_second_value_matrix = np.delete(np.mat(Flux_first_value_matrix), 0, 1)
    Flux_forth_value_matrix = (np.array(np.transpose(Flux_second_value_matrix))).astype(float)

    Fluxes_boundary_first_value_matrix = Fluxes_boundary.loc[Fluxes_boundary['Model_Reaction_ID'] == reaction_ID]  # match the reaction_ID to another dataframe ("Fluxes_boundary") to acquire the measurements of the lower and upper boundary values of flux
    Fluxes_boundary_second_value_matrix = np.delete(np.mat(Fluxes_boundary_first_value_matrix), 0, 1)
    j_obs_low1 = (Fluxes_boundary_second_value_matrix[:, 0:9]).astype(float)
    j_obs_up1 = (Fluxes_boundary_second_value_matrix[:, 9:18]).astype(float)
    return Flux_forth_value_matrix,  j_obs_low1, j_obs_up1


def thermodynamic(reaction_ID,Thermodynamic_data):
    '''
    :param reaction_ID: Reaction_ID in GEMs of S.cerevisiae (GEMs Yeast 7.6)
    :param Thermodynamic_data: a dataframe including all the thermodynamic data;
    :return: a thermodynamics data array; thermodynamics data = 1-e^(∆G/RT)
    '''
    thermodynamic_first_value_matrix=Thermodynamic_data.loc[(Thermodynamic_data['Model_Reaction_ID'] == reaction_ID)]
    thermodynamic_second_value_matrix=(np.array(np.delete(np.mat(thermodynamic_first_value_matrix),0,1))).astype(float)

    if np.isnan(thermodynamic_second_value_matrix).any() or np.any(thermodynamic_second_value_matrix == 0):    # the thermodynamic data constains zero or "Nan"
        deltG1=np.mat(np.ones(9))
    else:
        deltG1=np.abs(1-np.exp(thermodynamic_second_value_matrix/302.15/0.008314))
    deltG2=(np.transpose(deltG1)).astype(float)
    return deltG2

#
def logp(ux,lx,mu,sigma):
    '''
    Define the log likelihood function to use in Bayesian approach
    :param ux: the upper boundary of observed flux
    :param lx: the lower boundary of observed flux
    :param mu:  predicted flux by model
    :param sigma: mean square error obtained by calculating the average sum of squares of the difference between the flux predicted by the kinetic equation and the observed flux
    :return: a likelihood value of proposed parameters
    '''

    cdf_up = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(ux))
    cdf_low = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(lx))
    return pm.math.log(cdf_up-cdf_low)-pm.math.log(ux-lx)

def MLE_cal(flux_observed,flux_pos_mean,flux_pos_sd,array):
    """
    calculate the log maximum of the likelihood
    :param flux_observed: metabolic flux which is measured
    :param flux_pos_mean: metabolic flux predicted by the model
    :param flux_pos_sd:   mean squared error which was obtained by calculating the average sum of squares of the difference between the predicted flux and the measured flux
    :param array: an array to show how many conditions are considered in our model
    :return: an array including the log maximum of the likelihood
    """

    mle_matrix_1=np.square(
    (np.mat(flux_observed) - np.transpose(flux_pos_mean)) / (
                      1.41421 * np.transpose(flux_pos_sd)))
    mle_matrix_2=np.log(np.transpose(flux_pos_sd))
    mle_matrix_3=-1/ 2 * np.mat(np.full(shape=(array.shape[0], 1), fill_value=1.837877))
    return((mle_matrix_3-mle_matrix_2-mle_matrix_1).sum(axis=0))





def MCMC(deltG, flux_value, enzyme_value,j_obs_low, j_obs_up,reactants_log_MAP, proposed_allosteric_list_1 , proposed_inhibitors_list_1,
         proposed_activators_list_1, excel, possible_allosteric_list,possible_inhibitors_list, possible_activators_list,
         reactants_all_list, round, p_value,determined_coe,pearson_coe,MSE, best_allostery,lowermodel=None):
    '''
    Flow chart to perform MCMC sampling and acquire the plausible allosteric regulators:
    For each possible regulator, we perform MCMC sample first:
       for each iteration:
         (1) Propose a set of kinetic parameters drawn from the plausible prior distribution;
         (2) Calculate log predicted flux
         (3) Calculate the likelihood of kinetic parameter;
         (4) Evaluate the posterior probability of the kinetic parameter to determine whether the kinetic parameter we are drawn should be accepted or rejected;
       The above iteration will perform 120000 to acquire the posterior distribution of parameters. After verifying that all parameters converge to equivalent posterior distributions,
       A likelihood ratio test is performed to determine the candidate regulators from the possible regulators. Possible regulators with a P value < 0.05 were considered candidate regulators.
       Moreover, we also calculate the Pearson correlation coefficient and coefficient of determination by linear fitting of the measured flux with the output of the kinetic equation with plausible allosteric modulators across the nine experimental conditions.
       Regulators with a negative Pearson correlation coefficient or an increase in the determined coefficient of less than 0.15  are removed from the candidate regulators.

    Second, for each candidate regulator, we compared them by WAIC, and the candidate regulator with the lowest WAIC was considered a plausible regulator.

    :param deltG: thermodynamics data

    :param flux_value: metabolic fluxes

    :param enzyme_value:  enzyme abundance

    :param j_obs_low: the lower boundary flux

    :param j_obs_up: the upper boundary flux

    :param reactants_log_MAP:  a list including the log MAP of the parameters (only reactants without allosteric regulators, for the first time it is empty)

    :param proposed_allosteric_list_1: A list of all allosteric regulators

    :param proposed_inhibitors_list_1:  A list of all inhibitors

    :param proposed_activators_list_1: A list of all activators

    :param excel: An Excel that stores results from Bayesian inference

    :param possible_allosteric_list: A list including plausible allosteric regulators obtained from the latest Bayesian inference (for the first time, it is empty)

    :param possible_inhibitors_list: A list including the plausible inhibitors obtained from the latest Bayesian inference (for the first time, it is empty)

    :param possible_activators_list:  A list including the plausible activators obtained by the latest Bayesian inference (for the first time, it is empty)

    :param reactants_all_list: a list including all the reactants

    :param round: a variable that tells whether cooperative effect has been identified. round=1 and 2 means there is no cooperative effect, while the others indicate cooperative effect have identified

    :param p_value: a variable that determines the p value of the plausible regulators. For the first round, it is zero

    :param determined_coe: a variable that shows the determined coefficient between the flux predicted by the model with the plausible regulators and observed flux. For the first round, it is "Nan"

    :param pearson_coe: a variable that shows the Pearson correlation coefficient between the flux predicted by the model with the plausible regulators and observed flux. For the first round, it is "Nan"

    :param MSE:  a variable that shows the root mean squared error between the flux predicted by the model with the plausible regulators and observed flux. For the first round, it is "Nan"

    :param best_allostery: A list containing plausible allosteric regulators inferred through Bayesian inference (for the first time, it is empty)

    :param lowermodel: A parameter to determine the number of turns when performing the Bayesian inference (for the first time, it is "None")

    :return: A list containing all plausible allosteric regulators inferred by Bayesian inference and a database containing comparison results among all candidate regulators
    '''



    if lowermodel is None:  # this is the first round
        models, traces = OrderedDict(), OrderedDict()
        compareDict, nameConvDict = dict(), dict()

        try:
            with pm.Model() as models['reactants']: # DO MCMC sample for the generalized equation (an equation without possible allosteric regulators)

                # Log the generalized equation to construct a new dependent variable
                # generalized equation is j=kcat*e*(1-e^(∆G/RT))*(M/Km)^a
                # log-generalized equation will be log(j)=log(kcat)+log(e)+log(1-e^(∆G/RT))+a*log(M/Km)
                # construct a new dependent variable:log(o)= log(j)-log(e)-log(1-e^(∆G/RT)), then log(o)= log(kcat)+a*log(M/Km)

                flux_observed = np.array(np.log(flux_value) - np.log(enzyme_value) - np.log(deltG)).astype(float)   # construct a new dependent variable
                ln_j_obs_low = np.array(np.transpose(np.log(np.transpose(j_obs_low)) - np.log(enzyme_value) - np.log(deltG)))
                ln_j_obs_up = np.array(np.transpose(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value) - np.log(deltG)))
                dataframe1 = pd.DataFrame(flux_observed)

                # first: propose a set of kinetic parameters drawn from the plausible prior distribution
                reactants_value = r_mvalue(reactants_all_list, Metabolites)  # determine the value for all the reactants


                reactants_kinetic_order = pm.Uniform("_".join(reactants_all_list)+"_alpha", lower=0, upper=5,
                                                                      shape=reactants_value.shape[1])  # uniform distribution for the kinetic order
                possible_Kcatmin = (np.array(np.max(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value)))).astype(float)
                ln_kcat = pm.Uniform('ln_kcat', lower=possible_Kcatmin, upper=2.3026 + possible_Kcatmin)  # uniform distribution for the ln_kcat

                # second: Calculate log predicted flux
                reactants_multi_kinetic_order = pm.math.dot(reactants_value, reactants_kinetic_order)
                flux_predicted = pm.Deterministic('flux_P', ln_kcat +reactants_multi_kinetic_order)
                flux_mean_squared_error = pm.Deterministic('RMSE',pm.math.sqrt(((flux_observed - flux_predicted) ** 2).sum(
                    axis=0) / pm.math.abs_(reactants_value.shape[0])))

                # third: Calculate the likelihood of kinetic parameter
                flux_likelihood = pm.Normal.dist(flux_predicted, flux_mean_squared_error)

                j_obs = pm.DensityDist('j_obs', logp,
                                       observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low, 'mu': flux_predicted,
                                                 'sigma': flux_mean_squared_error},
                                       random=flux_likelihood.random)


                #fourth: Evaluate the posterior probability of the kinetic parameter to determine whether the kinetic parameter we are drawn should be accepted or rejected
                traces['reactants'] = pm.sample(10000, tune=50000, cores=2,
                                                start=pm.find_MAP(fmin=optimize.fmin_powell),
                                                progressbar=False)   # iteration are performed 120000

                # The maximum posterior distribution (MAP) is acquired from the posterior distribution
                traceplot = pm.summary(traces['reactants'])   # acquire the MAP
                print(traceplot)
                dataframe2 = pd.DataFrame(traceplot)

                # plot a posteriori distribution figure
                pm.traceplot(traces['reactants'])
                plt.savefig(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants_coefficient.eps',
                    dpi=600, format='eps')
                plt.close('all')

                # acquire the predicted flux according to the MAP of parameters
                flux_posterior_distribution_matrix = np.mat(
                    traceplot.iloc[
                    reactants_value.shape[1]+1:reactants_value.shape[1]+1 + reactants_value.shape[0],
                    0])
                dataframe3 = pd.DataFrame(flux_posterior_distribution_matrix)

                # acquire the mean squared error according to the predicted flux and observed flux
                flux_mean_squ_err_pos_dis_mat = np.sqrt(
                    ((flux_observed - np.array(np.transpose(flux_posterior_distribution_matrix))) ** 2).sum(axis=0) / (
                                                    np.abs(reactants_value.shape[0])))
                dataframe4 = pd.DataFrame(flux_mean_squ_err_pos_dis_mat)
                MSE=flux_mean_squ_err_pos_dis_mat[0]

                # do linear fitting between the predicted flux and observed flux
                flux_observed_matrix1 = np.squeeze(flux_observed)
                reactant_fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                reactant_fit_model.fit(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1)
                reactant_fit_model_coefficient = np.mat(
                    reactant_fit_model.score(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1))
                dataframe5 = pd.DataFrame(reactant_fit_model_coefficient)



                dataframe6 = pd.DataFrame(reactant_fit_model.coef_)
                fit_model_predicted = reactant_fit_model.predict(np.transpose(flux_posterior_distribution_matrix))

                # plot a linear fitting figure
                font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                plt.scatter(np.transpose(flux_posterior_distribution_matrix).tolist(),
                            flux_observed_matrix1.tolist(), c='g', marker='o', s=40)
                plt.plot(np.transpose(flux_posterior_distribution_matrix.tolist()), fit_model_predicted, c='r')
                plt.yticks(fontproperties='Arial', size=15)
                plt.xticks(fontproperties='Arial', size=15)
                plt.tick_params(width=2, direction='in')
                plt.xlabel("flux_posterior_distribution_value", font)
                plt.ylabel("Flux_observed", font)
                plt.savefig(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants-fitting_coefficient.eps',
                    dpi=600, format='eps')
                plt.close('all')

                # do Pearson correlation between the predicted flux and observed flux
                reactant_pearson_coe = pearsonr(np.squeeze(flux_observed),
                                                np.squeeze(np.transpose(np.array(flux_posterior_distribution_matrix))))
                pearson_coe=reactant_pearson_coe[0]
                dataframe7 = pd.DataFrame(reactant_pearson_coe)

                if reactant_pearson_coe[0]<0:
                    determined_coe=0
                else:
                    determined_coe = reactant_fit_model_coefficient[0, 0]

                # calculate the maximum likelihood of reactants
                reactants_log_MAP = MLE_cal(flux_observed, flux_posterior_distribution_matrix,
                                           flux_mean_squ_err_pos_dis_mat, reactants_value)
                dataframe8 = pd.DataFrame(reactants_log_MAP)

                writer = pd.ExcelWriter(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants_coefficient.xlsx')
                dataframe1.to_excel(writer, 'Flux_observed')
                dataframe2.to_excel(writer, 'reactants_coe_summary')
                dataframe3.to_excel(writer, 'flux_posterior_dis_data')
                dataframe4.to_excel(writer, 'flux_MSE_pos_dis_mat')
                dataframe5.to_excel(writer, 'fitting_coe')
                dataframe6.to_excel(writer, 'fitting_slope')
                dataframe7.to_excel(writer, 'Pearson_coefficient')
                dataframe8.to_excel(writer, 'reactant_log_MAP')
                writer.save()

        except RuntimeError:
            print('something error has happened, the program will start from next reference')
            return None

        # Do first model comparison. Since only reactants are available for the first round, the best allosteric regulators are assumed to be reactants.
        compareDict[models['reactants']] = traces['reactants']
        nameConvDict[models['reactants']] = 'reactants'
        compRst = pm.compare(compareDict)                     # compared all the possible allosteric regulators (only reactants for the first round)
        best_md_loc = compRst.index[compRst['rank'] == 0][0]
        best_allostery.append(nameConvDict[best_md_loc])     # add the reactants to the best allosteric regulators for the first round
        best_tc_loc = traces[nameConvDict[best_md_loc]]
        best_md = (best_md_loc, best_tc_loc)

        return MCMC(deltG, flux_value, enzyme_value,j_obs_low, j_obs_up,reactants_log_MAP, proposed_allosteric_list_1 , proposed_inhibitors_list_1,
         proposed_activators_list_1, excel, possible_allosteric_list,possible_inhibitors_list, possible_activators_list,
         reactants_all_list,round,p_value,determined_coe,pearson_coe,MSE, best_allostery, best_md)

    else: # this is not the first round (it also means that allosteric regulators should also be considered)
        assert best_allostery
        models, traces= OrderedDict(), OrderedDict()
        compareDict, nameConvDict, p_valueDict, correl_Dict, pearson_Dict,MSE_Dict= dict(), dict(),dict(),dict(),dict(),dict()

        for possible_allosteric_regulators in proposed_allosteric_list_1:    # judge allosteric regulator one by one
            print(possible_allosteric_regulators)

            try:
                with pm.Model() as models[possible_allosteric_regulators]:   # DO MCMC sample for the generalized equation + allosteric regulator

                    #Log the generalized equation to construct a new dependent variable
                    flux_observed = np.array(np.log(flux_value) - np.log(enzyme_value) - np.log(deltG)).astype(float)
                    ln_j_obs_low = np.array(np.transpose(np.log(np.transpose(j_obs_low)) - np.log(enzyme_value) - np.log(deltG)))
                    ln_j_obs_up = np.array(np.transpose(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value) - np.log(deltG)))
                    dataframe1 = pd.DataFrame(flux_observed)

                    # first: propose a set of kinetic parameters drawn from the plausible prior distribution
                    reactants_value = r_mvalue(reactants_all_list,
                                               Metabolites)  # determine the value for all the reactants

                    reactants_kinetic_order = pm.Uniform("_".join(reactants_all_list)+"_"+possible_allosteric_regulators+"alpha", lower=0, upper=5,
                                                         shape=reactants_value.shape[
                                                             1])  # uniform distribution for the kinetic order

                    reactants_multi_kinetic_order = pm.math.dot(reactants_value, reactants_kinetic_order)

                    additional_parameter=0                            # a variable that indicates how many parameters are added

                    current_possible_allosteric_value_matrix = r_savalue(possible_allosteric_regulators,
                                                                             Metabolites)             # determine the value for possible allosteric regulators

                    current_possible_allosteric_median = np.median(current_possible_allosteric_value_matrix, axis=0)
                    current_log_Km_value = pm.Uniform(possible_allosteric_regulators + '_c_log_Km',
                                                      lower=-15 + current_possible_allosteric_median,
                                                      upper=15 + current_possible_allosteric_median,
                                                      shape=current_possible_allosteric_value_matrix.shape[1])             # uniform distribution for the log Km value


                    # judge the function of the possible allosteric regulators according to the prior knowledge
                    if possible_allosteric_regulators in proposed_inhibitors_list_1 and possible_allosteric_regulators in proposed_activators_list_1:
                        current_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_c_alpha', lower=-5,
                                                           upper=5,
                                                           shape=current_possible_allosteric_value_matrix.shape[1])
                    elif possible_allosteric_regulators in proposed_inhibitors_list_1 and possible_allosteric_regulators not in proposed_activators_list_1:
                        current_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_c_alpha', lower=-5,
                                                           upper=0,
                                                           shape=current_possible_allosteric_value_matrix.shape[1])

                    elif possible_allosteric_regulators not in proposed_inhibitors_list_1 and possible_allosteric_regulators in proposed_activators_list_1:
                        current_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_c_alpha', lower=0,
                                                           upper=5,
                                                           shape=current_possible_allosteric_value_matrix.shape[1])

                    current_allostery_multi_kinetic_order = pm.math.dot(
                        current_possible_allosteric_value_matrix - current_log_Km_value,
                        current_kinetic_order)

                    additional_parameter=additional_parameter+2

                    # judge whether the plausible allosteric regulators exist after multiple rounds
                    if len(possible_allosteric_list) == 0:
                        previous_allostery_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                        additional_parameter = additional_parameter

                    else:
                        possible_allosteric_value_matrix = r_avalue(possible_allosteric_list, Metabolites)
                        possible_allosteric_value_matrix_median = np.median(possible_allosteric_value_matrix, axis=0)
                        allsoteric_log_Km_value = pm.Uniform(possible_allosteric_regulators + '_al_log_Km',
                                                             lower=-15 + possible_allosteric_value_matrix_median,
                                                             upper=15 + possible_allosteric_value_matrix_median,
                                                             shape=len(possible_allosteric_list))     # uniform distribution for Km

                        allosteric_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_al_alpha',
                                                              lower=-5, upper=5, shape=len(possible_allosteric_list))   # uniform distribution for kinetic order
                        previous_allostery_multi_kinetic_order = pm.math.dot(possible_allosteric_value_matrix - allsoteric_log_Km_value,
                                                                 allosteric_kinetic_order)
                        additional_parameter =  additional_parameter + len(possible_allosteric_list)*2


                    # judge whether the plausible inhibitors exist after multiple rounds
                    if len(possible_inhibitors_list) == 0:
                        previous_inhibitors_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                        additional_parameter = additional_parameter
                    else:
                        possible_inhibitors_value_matrix = r_avalue(possible_inhibitors_list, Metabolites)

                        possible_inhibitors_value_matrix_median = np.median(possible_inhibitors_value_matrix, axis=0)
                        inhibitors_log_Km_value = pm.Uniform(possible_allosteric_regulators + '_i_log_Km',
                                                             lower=-15 + possible_inhibitors_value_matrix_median,
                                                             upper=15 + possible_inhibitors_value_matrix_median,
                                                             shape=len(possible_inhibitors_list))     # uniform distribution for Km
                        inhibitors_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_i_alpha',
                                                              lower=-5, upper=0, shape=len(possible_inhibitors_list))   # uniform distribution for kinetic order
                        previous_inhibitors_multi_kinetic_order = pm.math.dot(possible_inhibitors_value_matrix - inhibitors_log_Km_value,
                                                                 inhibitors_kinetic_order)
                        additional_parameter =  additional_parameter + len(possible_inhibitors_list)*2

                    # judge whether the plausible activators exist after multiple rounds
                    if len(possible_activators_list) == 0:
                        previous_activators_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                        additional_parameter = additional_parameter
                    else:
                        possible_activators_value_matrix = r_avalue(possible_activators_list, Metabolites)

                        possible_activators_value_matrix_median = np.median(possible_activators_value_matrix, axis=0)
                        activators_log_Km_value = pm.Uniform(possible_allosteric_regulators + '_ac_log_Km',
                                                             lower=-15 + possible_activators_value_matrix_median,
                                                             upper=15 + possible_activators_value_matrix_median,
                                                             shape=len(possible_activators_list))     # uniform distribution for Km
                        activators_kinetic_order = pm.Uniform(possible_allosteric_regulators + '_ac_alpha',
                                                              lower=0, upper=5, shape=len(possible_activators_list))   # uniform distribution for kinetic order
                        previous_activators_multi_kinetic_order = pm.math.dot(possible_activators_value_matrix - activators_log_Km_value,
                                                                 activators_kinetic_order)
                        additional_parameter =  additional_parameter + len(possible_activators_list)*2

                    possible_Kcatmin = (np.array(np.max(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value)))).astype(
                        float)
                    ln_kcat = pm.Uniform('ln_kcat', lower=possible_Kcatmin, upper=2.3026 + possible_Kcatmin)     # uniform distribution for kcat



                    # second: Calculate log predicted flux
                    flux_predicted = pm.Deterministic(possible_allosteric_regulators + '_flux',
                                                      ln_kcat + reactants_multi_kinetic_order + current_allostery_multi_kinetic_order + previous_allostery_multi_kinetic_order + previous_inhibitors_multi_kinetic_order + previous_activators_multi_kinetic_order)

                    flux_mean_squared_error = pm.Deterministic(possible_allosteric_regulators + '_RMSE',pm.math.sqrt(((flux_observed - flux_predicted) ** 2).sum(
                        axis=0) / pm.math.abs_(reactants_value.shape[0])))


                    # third: Calculate the likelihood of kinetic parameter
                    flux_likelihood = pm.Normal.dist(flux_predicted, flux_mean_squared_error)
                    j_obs = pm.DensityDist(possible_allosteric_regulators + 'j_obs', logp,
                                           observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low, 'mu': flux_predicted,
                                                     'sigma': flux_mean_squared_error},
                                           random=flux_likelihood.random)

                    # fourth: Evaluate the posterior probability of the kinetic parameter to determine whether the kinetic parameter we are drawn should be accepted or rejected
                    traces[possible_allosteric_regulators] = pm.sample(10000, tune=80000, cores=2,
                                                                     start=pm.find_MAP(fmin=optimize.fmin_powell),
                                                                     progressbar=False)


                    # The maximum posterior distribution (MAP) is acquired from the posterior distribution
                    traceplot = pm.summary(traces[possible_allosteric_regulators])    # acquire the MAP
                    print(traceplot)
                    dataframe2 = pd.DataFrame(traceplot)


                    # plot a posteriori distribution figure
                    pm.traceplot(traces[possible_allosteric_regulators])
                    plt.savefig(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allostery) + possible_allosteric_regulators + '_coefficient.eps',
                        dpi=600, format='eps')
                    plt.close('all')

                    # acquire the predicted flux according to the MAP of parameters
                    flux_posterior_distribution_matrix = np.mat(
                        traceplot.iloc[
                        len(reactants_all_list)+additional_parameter+1: len(reactants_all_list)+additional_parameter+1+reactants_value.shape[0],0])
                    dataframe3 = pd.DataFrame(flux_posterior_distribution_matrix)

                    # acquire the mean squared error according to the predicted flux and observed flux
                    flux_mean_squ_err_pos_dis_mat = np.sqrt(
                        ((flux_observed - np.array(np.transpose(flux_posterior_distribution_matrix))) ** 2).sum(
                            axis=0) / (np.abs(reactants_value.shape[0])))
                    dataframe4 = pd.DataFrame(flux_mean_squ_err_pos_dis_mat)


                    # do linear fitting between the predicted flux and observed flux
                    flux_observed_matrix1 = np.squeeze(flux_observed)
                    allostery_fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                    allostery_fit_model.fit(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1)
                    allostery_fit_model_coefficient = np.mat(
                        allostery_fit_model.score(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1))
                    dataframe5 = pd.DataFrame(allostery_fit_model_coefficient)
                    dataframe6 = pd.DataFrame(allostery_fit_model.coef_)
                    allostery_fit_model_predicted = allostery_fit_model.predict(np.transpose(flux_posterior_distribution_matrix))

                    # plot a linear fitting figure
                    font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                    plt.scatter(np.transpose(flux_posterior_distribution_matrix).tolist(),
                                flux_observed_matrix1.tolist(), c='g', marker='o', s=40)
                    plt.plot(np.transpose(flux_posterior_distribution_matrix.tolist()), allostery_fit_model_predicted, c='r')
                    plt.yticks(fontproperties='Arial', size=15)
                    plt.xticks(fontproperties='Arial', size=15)
                    plt.tick_params(width=2, direction='in')
                    plt.xlabel("flux_posterior_distribution_value", font)
                    plt.ylabel("Flux_observed", font)
                    plt.savefig(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allostery) + possible_allosteric_regulators + '_fitting_curve.eps',
                        dpi=600, format='eps')
                    plt.close('all')

                    # do Pearson correlation between the predicted flux and observed flux
                    allostery_pearson_coe = pearsonr(np.squeeze(flux_observed),
                                                    np.squeeze(
                                                        np.transpose(np.array(flux_posterior_distribution_matrix))))
                    dataframe7 = pd.DataFrame(allostery_pearson_coe)

                    # calculate the maximum likelihood of allostery
                    allostery_log_MAP = MLE_cal(flux_observed, flux_posterior_distribution_matrix,
                                               flux_mean_squ_err_pos_dis_mat, reactants_value)
                    dataframe8 = pd.DataFrame(allostery_log_MAP)


                    # Do likelihood ratio test for each allosteric regulator
                    allostery_p_value = chi2.sf(2 * (allostery_log_MAP - reactants_log_MAP), additional_parameter)
                    dataframe9 = pd.DataFrame(allostery_p_value)

                    writer = pd.ExcelWriter(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allostery) + possible_allosteric_regulators + '_coefficient.xlsx')
                    dataframe1.to_excel(writer, 'Flux_observed')
                    dataframe2.to_excel(writer, 'reactants_coe_summary')
                    dataframe3.to_excel(writer, 'flux_posterior_dis_data')
                    dataframe4.to_excel(writer, 'flux_MSE_pos_dis_mat')
                    dataframe5.to_excel(writer, 'fitting_coe')
                    dataframe6.to_excel(writer, 'fitting_slope')
                    dataframe7.to_excel(writer, 'Pearson_coefficient')
                    dataframe8.to_excel(writer, 'allostery_log_MAP')
                    dataframe9.to_excel(writer, 'allostery_p_value')
                    writer.save()
            except RuntimeError:
                print('something error has happened, the program will start from next reference')
                continue





            if round == 1:    # this is the first round. We should consider these limitations as follows:

                if allostery_p_value[0,0] < 0.05 and allostery_pearson_coe[0] > 0 and (allostery_fit_model_coefficient[0,0] - determined_coe)> 0.15:       # Screening of candidate allostery
                    compareDict[models[possible_allosteric_regulators]] = traces[possible_allosteric_regulators]
                    nameConvDict[models[possible_allosteric_regulators]] = possible_allosteric_regulators
                    p_valueDict[models[possible_allosteric_regulators]] = allostery_p_value[0, 0]
                    correl_Dict[models[possible_allosteric_regulators]] = allostery_fit_model_coefficient[0,0]
                    pearson_Dict[models[possible_allosteric_regulators]] = allostery_pearson_coe[0]
                    MSE_Dict[models[possible_allosteric_regulators]] = flux_mean_squ_err_pos_dis_mat[0]

            else:    # for other rounds, we only consider this limitation as follows
                if allostery_p_value[0,0] < 0.05:
                    compareDict[models[possible_allosteric_regulators]] = traces[possible_allosteric_regulators]
                    nameConvDict[models[possible_allosteric_regulators]] = possible_allosteric_regulators
                    p_valueDict[models[possible_allosteric_regulators]] = allostery_p_value[0, 0]
                    correl_Dict[models[possible_allosteric_regulators]] = allostery_fit_model_coefficient[0,0]
                    pearson_Dict[models[possible_allosteric_regulators]]= allostery_pearson_coe[0]
                    MSE_Dict[models[possible_allosteric_regulators]] = flux_mean_squ_err_pos_dis_mat[0]

        compareDict[lowermodel[0]] = lowermodel[1]
        if compareDict:
            nameConvDict_dataframe = pd.DataFrame(nameConvDict, index=[0])
            p_valueDict_dataframe = pd.DataFrame(p_valueDict, index=[0])
            correl_Dict_dataframe = pd.DataFrame(correl_Dict, index=[0])
            pearson_Dict_dataframe= pd.DataFrame(pearson_Dict, index=[0])
            MSE_Dict_dataframe = pd.DataFrame(MSE_Dict, index=[0])

            # construct a dataframe to include all the results
            connect_dataframe_1 = pd.concat(
                [nameConvDict_dataframe, p_valueDict_dataframe, correl_Dict_dataframe, pearson_Dict_dataframe,MSE_Dict_dataframe],
                axis=0)
            connect_dataframe_1.index = ['possible_regulator', 'p_value', 'determined_coefficient', 'Pearson_coefficient','mean_square_error']
            connect_dataframe_2 = pd.DataFrame(connect_dataframe_1.values.T, index=connect_dataframe_1.columns,
                                               columns=connect_dataframe_1.index)
            best_results_series = pd.Series(
                {"possible_regulator": best_allostery, "p_value": p_value, 'determined_coefficient': determined_coe, 'Pearson_coefficient':pearson_coe,
                 'mean_square_error': MSE}, name=lowermodel[0])
            connect_dataframe_3 = connect_dataframe_2.append(best_results_series)

            compRst_1 = pm.compare(compareDict)              # DO comparison among all the candidate regulators
            compRst_2 = pd.concat([compRst_1, connect_dataframe_3], axis=1)
            print(compRst_2)
            connect_dataframe_3.to_excel(excel, "-".join(best_allostery) + "-" + 'all_name')
            compRst_2.to_excel(excel, "-".join(best_allostery) + "-" + 'model_compare')
            best_md_loc = compRst_1.index[compRst_1['rank'] == 0][0]

            if best_md_loc == lowermodel[0]:    # there are no regulators that best supported the reaction
                print('Finally, found the best model is\033[1;31;43m', best_allostery, '\033[0m')
                return best_allostery

            else:                # cooperative regulators
                round=round+1
                p_value = compRst_2.iloc[0].at['p_value']
                determined_coe = compRst_2.iloc[0].at['determined_coefficient']
                pearson_coe=compRst_2.iloc[0].at['Pearson_coefficient']
                MSE = compRst_2.iloc[0].at['mean_square_error']
                best_tc_loc = traces[nameConvDict[best_md_loc]]
                best_md = (best_md_loc, best_tc_loc)
                best_allostery.append(nameConvDict[best_md_loc])
                proposed_allosteric_list_1.remove(nameConvDict[best_md_loc])

                if nameConvDict[best_md_loc] in proposed_inhibitors_list_1 and nameConvDict[
                    best_md_loc] in proposed_activators_list_1:
                    possible_allosteric_list = possible_allosteric_list + nameConvDict[best_md_loc].split(' ')

                elif nameConvDict[best_md_loc] in proposed_inhibitors_list_1 and nameConvDict[
                    best_md_loc] not in proposed_activators_list_1:
                    possible_inhibitors_list = possible_inhibitors_list + nameConvDict[best_md_loc].split(' ')

                elif nameConvDict[best_md_loc] in proposed_activators_list_1 and nameConvDict[
                    best_md_loc] not in proposed_inhibitors_list_1:
                    possible_activators_list = possible_activators_list + nameConvDict[best_md_loc].split(' ')

                return MCMC(deltG, flux_value, enzyme_value, j_obs_low, j_obs_up, reactants_log_MAP,
                            proposed_allosteric_list_1, proposed_inhibitors_list_1,
                            proposed_activators_list_1, excel, possible_allosteric_list, possible_inhibitors_list,
                            possible_activators_list,
                            reactants_all_list, round, p_value,
                            determined_coe, pearson_coe,MSE, best_allostery, best_md)

        else:
            return None



# acquire data from excel ("Hierachical regulation analysis-CCMS(openmebius)")
Reactions=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Reaction')
# a dataframe including all the reactions (including reaction ID, Enzyme ID, reactants ID, and Km value
Metabolites=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='metabolites_mM')
# a dataframe including all the metabolite IDs and its values
Enzymes=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='enzyme_umolgDCW_mean')
# a dataframe containing all the enzyme ID and its values
Fluxes=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Flux_mean')
# a dataframe containing all the flux ID and its values
Fluxes_boundary=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='FVA_Fluxes_mean')
# a dataframe containing all the flux ID and its lower and upper boundary flux values
Enzyme_Transform=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='enzyme transform')
# a dataframe containing all the enzyme ID and its inhibitors and activators
Thermodynamic_data=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Thermodynamic Data')
# a dataframe containing all the reaction ID and its thermodynamic data



for step in range(0,len(Reactions)):
    reaction_ID=Reactions.iloc[step,0]
    print(reaction_ID)           # show which reaction is executing

    reactants_dataframe=r_rdataframe(reaction_ID,step,Reactions)   # acquire reactants dataframe

    enzyme_dataframe =r_edataframe(step,Reactions,reactants_dataframe,reaction_ID)  # acquire enzyme dataframe


    proposed_enzyme_ID= list(enzyme_dataframe['Enzyme_ID'][enzyme_dataframe['Measured_Value'] == 'True'])  # Collect enzymes whose 'Measured_Value' is "True" in the enzyme dataframe to make a list

    enzyme_value=r_evalue(proposed_enzyme_ID,Enzymes)    # acquire an enzyme abundance array

    flux_value, j_obs_low, j_obs_up = r_fvalue(reaction_ID, Fluxes, Fluxes_boundary)   # acquire a flux array, an array containing lower boundary of flux values, and an array including upper boundary of flux values

    deltG = thermodynamic(reaction_ID,Thermodynamic_data)   # acquire a thermodynamics data array

    inhibitors_dataframe, activitors_dataframe = a_dataframe(reaction_ID, proposed_enzyme_ID, Enzyme_Transform)   # acquire an inhibitors_dataframe and activitors_dataframe

    proposed_inhibitors_ID = list(
        inhibitors_dataframe['Inhibitors_ID'][inhibitors_dataframe['Inhibitors_Value'] == 'True'])     # Collect inhibitors whose 'Inhibitors_Value' is "True" in the inhibitors_dataframe to make a list

    proposed_activators_ID = list(
        activitors_dataframe['Activators_ID'][activitors_dataframe['Activators_Value'] == 'True'])     # Collect activators whose 'Activators_Value' is "True" in the activitors_dataframe to make a list

    reactants_all_list,reactants_noallostery_list, reactants_inhibitors_list, reactants_activators_list, proposed_inhibitors_list, proposed_activators_list=met_judge(reactants_dataframe, proposed_inhibitors_ID,proposed_activators_ID)
    # compare the reactants list to inhibitor list and activator list to determine whether the reactants are allosteric regulators and form six lists


    proposed_inhibitors_list_1=reactants_inhibitors_list+proposed_inhibitors_list   # collect all the inhibitors (including the reactants whose function are inhibitory according to the database

    proposed_activators_list_1=reactants_activators_list+proposed_activators_list   # collect all the activators (including the reactants whose function are activatory according to the database

    reactants_allosteric_list_1=list(set(reactants_inhibitors_list+reactants_activators_list))    # collect all the reactants whose function are allostery


    proposed_allosteric_list_1 = list(set(proposed_inhibitors_list_1 + proposed_activators_list_1)) # collect all the inhibitors and activators
    print(proposed_allosteric_list_1)

    if np.all(flux_value == 0) or np.isnan(flux_value).any() or np.isnan(j_obs_up).any() or np.isnan(
            j_obs_low).any():                 # means no fluxes are estimated for the reaction

        print(reaction_ID, "has no detective flux, next reaction will start soon")
        continue

    else:
        if np.isnan(enzyme_value).any() or np.all(enzyme_value == 0):    # means no enzyme values are measured for the reaction
            pass
        else:

            os.makedirs('debug/Bayesinference/Reaction/parameters/' + reaction_ID)     # Create a folder to store the output of the Bayesian approach (the posterior distribution of parameters, the MAP of the parameters, the flux predicted by model,etc.)
            os.makedirs('debug/Bayesinference/Reaction/statistic/' + reaction_ID)      # Create a folder to store the output of the Bayesian approach (reults of all the model comparisons)
            excel = pd.ExcelWriter(
                'debug/Bayesinference/Reaction/statistic/' + reaction_ID + '/-results_statistics .xlsx')

            best_allostery = []        # create a list to include the plausible allosteric regulators inferred by Bayesian inference

            
            possible_allosteric_list = list()    # Create a list to include the plausible allosteric regulator obtained by the last Bayesian inference

            possible_inhibitors_list = list()    # Create a list to include the plausible inhibitors obtained by the last Bayesian inference

            possible_activators_list = list()    # Create a list to include the plausible activators obtained by the last Bayesian inference

            reactants_log_MAP = []        # create a list to include the log MAP of the parameters (only reactants without allosteric regulators)

            round=1       # this is the first round to determine whether the possible regulator is candidate regulator

            p_value = 0      # we think each reactant which has no allosteric function is significantly improve the model fitting

            determined_coe = 'Nan'    # before doing MCMC sampling, we think the determined coefficient between the flux predicted by generalized model (without allosteric regulators) and observed flux is "Nan"

            pearson_coe ="Nan"   # before doing MCMC sampling, we think the Pearson coefficient between the flux predicted by generalized model (without allosteric regulators) and observed flux is "Nan"

            MSE = 'Nan'      # before doing MCMC sampling, we think the MSE between flux predicted by generalized model (without allosteric regulators) and observed flux is "Nan"

            # DO MCMC sample to acquire the posterior distribution of parameters and the plausible allosteric regulators
            best_allostery = MCMC(deltG, flux_value, enzyme_value,
                                               j_obs_low, j_obs_up,reactants_log_MAP,
                                               proposed_allosteric_list_1 , proposed_inhibitors_list_1,
                                               proposed_activators_list_1, excel, possible_allosteric_list,
                                               possible_inhibitors_list, possible_activators_list, reactants_all_list,round,p_value,determined_coe,pearson_coe,MSE,best_allostery,
                                               lowermodel=None)

            best_allostery_dataframe = pd.DataFrame(best_allostery)
            best_allostery_dataframe.to_excel(excel, 'best_allostery')
            excel.save()

















