"""
After downloading the USPTO dataset from ORD, this script will extract the data and write it to a pickle file.

Instructions:
1) Create a folder called "data/USPTO" in the same directory as this script 
2) Download the USPTO data from ORD
2.1) While inside USPTO: git clone https://github.com/open-reaction-database/ord-data and you'll find the data in ord-data/data/
2.2) You'll notice that the data is split into folders, each containing a number of ord files. They are batched by year.
3) python USPTO_extraction.py True
4) Solvents list originally from https://github.com/sustainable-processes/vle_prediction/blob/master/data/cosmo/solvent_descriptors.csv (I've added methanol (CO), and also added 'ClP(Cl)Cl' and 'ClS(Cl)=O' as smiles strings)

# Output:
1) A pickle file with the cleaned data for each folder of uspto data. NB: Temp always in C, time always in hours
"""

# Import modules
#import ord_schema
from ord_schema import message_helpers#, validations
from ord_schema.proto import dataset_pb2


import pandas as pd
import numpy as np
import os

import pickle
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime

from rdkit import Chem

from tqdm import tqdm
import os
import sys



"""
# Disables RDKit whiny logging.
# """
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')
from rdkit.rdBase import BlockLogs



class OrdToPickle():
    """
    Read in an ord file, check if it contains USPTO data, and then:
    1) Extract all the relevant data (raw): reactants, products, catalysts, reagents, yields, temp, time
    2) Canonicalise all the molecules
    3) Write to a pickle file
    """

    def __init__(self, ord_file_path, merge_cat_and_reag, replacements_dict, solvents_set):
        self.ord_file_path = ord_file_path
        self.data = message_helpers.load_message(self.ord_file_path, dataset_pb2.Dataset)
        self.filename = self.data.name
        self.names_list = []
        self.merge_cat_solv_reag = merge_cat_and_reag 
        self.replacements_dict = replacements_dict
        self.solvents_set = solvents_set
        
    def find_smiles(self, identifiers):
        block = BlockLogs()
        for i in identifiers:
            if i.type == 2:
                smiles = self.clean_smiles(i.value)
                return smiles
        for ii in identifiers: #if there's no smiles, return the name
            if ii.type == 6:
                name = ii.value
                self.names_list += [name]
                return name
        return None

    def clean_mapped_smiles(self, smiles):
        block = BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            m = Chem.MolFromSmiles(smiles)
            for atom in m.GetAtoms():
                atom.SetAtomMapNum(0)
            cleaned_smiles = Chem.MolToSmiles(m)
            return cleaned_smiles
        except AttributeError:
            self.names_list += [smiles]
            return smiles

    def clean_smiles(self, smiles):
        block = BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            cleaned_smiles = Chem.CanonSmiles(smiles)
            return cleaned_smiles
        except:
            self.names_list += [smiles]
            return smiles

    #its probably a lot faster to sanitise the whole thing at the end
    # NB: And create a hash map/dict



    def build_rxn_lists(self):
        mapped_rxn_all = []
        reactants_all = []
        reagents_all = []
        agents_all = []
        products_all = []
        solvents_all = []
        catalysts_all = []

        temperature_all = []

        rxn_times_all = []

        yields_all = []

        for i in range(len(self.data.reactions)):
            rxn = self.data.reactions[i]
            # handle rxn inputs: reactants, reagents etc
            reactants = []
            reagents = []
            solvents = []
            catalysts = []
            marked_products = []
            mapped_products = []
            products = []
            not_mapped_products = []
            
            temperatures = []

            rxn_times = []

            yields = []
            mapped_yields = []
            

            #if reaction has been mapped, get reactant and product from the mapped reaction
            #Actually, we should only extract data from reactions that have been mapped
            is_mapped = self.data.reactions[i].identifiers[0].is_mapped
            if is_mapped:
                mapped_rxn_extended_smiles = self.data.reactions[i].identifiers[0].value
                mapped_rxn = mapped_rxn_extended_smiles.split(' ')[0]

                reactant, reagent, mapped_product = mapped_rxn.split('>')

                for r in reactant.split('.'):
                    if '[' in r and ']' in r and ':' in r:
                        reactants += [r]
                    else:
                        reagents += [r]

                reagents += [r for r in reagent.split('.')]

                for p in mapped_product.split('.'):
                    if '[' in p and ']' in p and ':' in p:
                        mapped_products += [p]
                        
                    else:
                        not_mapped_products += [p]


                # inputs
                for key in rxn.inputs: #these are the keys in the 'dict' style data struct
                    try:
                        components = rxn.inputs[key].components
                        for component in components:
                            rxn_role = component.reaction_role #rxn role
                            identifiers = component.identifiers
                            smiles = self.find_smiles(identifiers)
                            if rxn_role == 1: #reactant
                                #reactants += [smiles]
                                # we already added reactants from mapped rxn
                                # So instead I'll add it to the reagents list
                                # A lot of the reagents seem to have been misclassified as reactants
                                # I just need to remember to remove reagents that already appear as reactants
                                #   when I do cleaning

                                reagents += [r for r in smiles.split('.')]
                            elif rxn_role ==2: #reagent
                                reagents += [r for r in smiles.split('.')]
                            elif rxn_role ==3: #solvent
                                # solvents += [smiles] # I initially tried to let the solvents stay together, but actually it's better to split them up
                                # Examples like CO.O should probably be split into CO and O
                                solvents += [r for r in smiles.split('.')]
                            elif rxn_role ==4: #catalyst
                                # catalysts += [smiles] same as solvents
                                catalysts += [r for r in smiles.split('.')]
                            elif rxn_role in [5,6,7]: #workup, internal standard, authentic standard. don't care about these
                                continue
                            # elif rxn_role ==8: #product
                            #     #products += [smiles]
                            # there are no products recorded in rxn_role == 8, they're all stored in "outcomes"
                    except IndexError:
                        #print(i, key )
                        continue

                # temperature
                try:
                    # first look for the temperature as a number
                    temp_unit = rxn.conditions.temperature.setpoint.units
                        
                    if temp_unit == 1: #celcius
                        temperatures +=[rxn.conditions.temperature.setpoint.units]
                        
                    elif temp_unit == 2: #fahrenheit
                        f = rxn.conditions.temperature.setpoint.units
                        c = (f-32)*5/9
                        temperatures +=[c]
                        
                    elif temp_unit == 3: #kelvin
                        k = rxn.conditions.temperature.setpoint.units
                        c = k - 273.15
                        temperatures +=[c]
                    elif temp_unit == 0:
                        if temp_unit == 0: #unspecified
                            #instead of using the setpoint, use the control type
                            #temperatures are in celcius
                            temp_control_type = rxn.conditions.temperature.control.type
                            if temp_control_type == 2: #AMBIENT
                                temperatures +=[25]
                            elif temp_control_type == 6: #ICE_BATH
                                temperatures +=[0]
                            elif temp_control_type == 9: #DRY_ICE_BATH
                                temperatures +=[-79]
                            elif temp_control_type == 11: #LIQUID_NITROGEN
                                temperatures +=[-196]   
                except IndexError:
                    continue
                    

                #rxn time
                try:
                    if rxn.outcomes[0].reaction_time.units == 1: #hour
                        rxn_times += [rxn.outcomes[0].reaction_time.value]
                    elif rxn.outcomes[0].reaction_time.units == 3: #seconds
                        s = rxn.outcomes[0].reaction_time.value
                        h = s/3600
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 2: #minutes
                        m =  rxn.outcomes[0].reaction_time.value
                        h = m/60
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 4: #day
                        d = rxn.outcomes[0].reaction_time.value
                        h = d*24
                        rxn_times += [h]
                except IndexError:
                    continue

                # products & yield
                products_obj = rxn.outcomes[0].products
                y1 = np.nan
                y2 = np.nan
                for marked_product in products_obj:
                    try:
                        identifiers = marked_product.identifiers
                        product_smiles = self.find_smiles(identifiers)
                        measurements = marked_product.measurements
                        for measurement in measurements:
                            if measurement.details =="PERCENTYIELD":
                                y1 = measurement.percentage.value
                            elif measurement.details =="CALCULATEDPERCENTYIELD":
                                y2 = measurement.percentage.value
                        #marked_products += [(product_smiles, y1, y2)]
                        marked_products += [product_smiles]
                        if y1 == y1:
                            yields += [y1]
                        elif y2==y2:
                            yields +=[y2]
                        else:
                            yields += [np.nan]
                    except IndexError:
                        continue
            
            #clean the smiles

            #remove reagents that are integers
            #reagents = [x for x in reagents if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
            # I'm assuming there are no negative integers
            reagents = [x for x in reagents if not (x.isdigit())]

            reactants = [self.clean_mapped_smiles(smi) for smi in reactants]
            
            reagents = [self.clean_smiles(smi) for smi in reagents]
            solvents = [self.clean_smiles(smi) for smi in solvents]
            catalysts = [self.clean_smiles(smi) for smi in catalysts]

            # Apply the replacements_dict to the reagents, solvents, and catalysts
            reagents  = list((pd.Series(reagents, dtype=pd.StringDtype())).replace(self.replacements_dict))
            solvents  = list((pd.Series(solvents, dtype=pd.StringDtype())).replace(self.replacements_dict))
            catalysts = list((pd.Series(catalysts, dtype=pd.StringDtype())).replace(self.replacements_dict))
            
            # Split out any instances of a . in the smiles strings
            reagents = [substring for reagent in reagents for substring in reagent.split('.')]
            solvents = [substring for solvent in solvents for substring in solvent.split('.')]
            catalysts = [substring for catalyst in catalysts for substring in catalyst.split('.')]
            
            
            mapped_rxn_all += [mapped_rxn]
            reactants_all += [reactants]
            
            
            
            if self.merge_cat_solv_reag == True:
                agents = catalysts + solvents + reagents # merge the solvents, reagents, and catalysts into one list
                agents_set = set(agents) # this includes the solvnts
                
                # build two new lists, one with the solvents, and one with the reagents+catalysts
                # Create a new set of solvents from agents_set
                solvents = agents_set.intersection(self.solvents_set)

                # Remove the solvents from agents_set
                agents = agents_set.difference(solvents)
                
                     
                # I think we should add some ordering to the agents
                # What if we first order them alphabetically, and afterwards by putting the metals first in the list
                
                agents = sorted(list(agents))
                solvents = sorted(list(solvents))
           

                # Ideally we'd order the agents, so we have the catalysts (metal centre) first, then the ligands, then the bases and finally any reagents
                # We don't have a list of catalysts, and it's not straight forward to figure out if something is a catalyst or not (both chemically and computationally)
                # Instead, let's move all agents that contain a metal centre to the front of the list
                
                metals = [
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv'
]
                agents = [agent for agent in agents if any(metal in agent for metal in metals)] + [agent for agent in agents if not any(metal in agent for metal in metals)]
 
                agents_all += [agents]
                solvents_all += [solvents]
            else:
                solvents_all += [list(set(solvents))]
                reagents_all += [list(set(reagents))]
                catalysts_all += [list(set(catalysts))]
                
            
            temperature_all = [temperatures]

            rxn_times_all += [rxn_times]


            # products logic
            # handle the products
            # for each row, I will trust the mapped product more
            # loop over the mapped products, and if the mapped product exists in the marked product
            # add the yields, else simply add smiles and np.nan

            # canon and remove mapped info from products
            mapped_p_clean = [self.clean_mapped_smiles(p) for p in mapped_products]
            marked_p_clean = [self.clean_smiles(p) for p in marked_products]
            # What if there's a marked product that only has the correct name, but not the smiles?



            for mapped_p in mapped_p_clean:
                added = False
                for ii, marked_p in enumerate(marked_p_clean):
                    if mapped_p == marked_p and mapped_p not in products:
                        products+= [mapped_p]
                        mapped_yields += [yields[ii]]
                        added = True
                        break

                if not added and mapped_p not in products:
                    products+= [mapped_p]
                    mapped_yields += [np.nan]
            

            products_all += [products] 
            yields_all +=[mapped_yields]


        
        return mapped_rxn_all, reactants_all, agents_all, reagents_all, solvents_all, catalysts_all, temperature_all, rxn_times_all, products_all, yields_all

    # create the column headers for the df
    def create_column_headers(self, df, base_string):
        column_headers = []
        for i in range(len(df.columns)):
            column_headers += [base_string+str(i)]
        return column_headers
    
    def build_full_df(self):
        headers = ['mapped_rxn_', 'reactant_', 'agent_', 'reagent_',  'solvent_', 'catalyst_', 'temperature_', 'rxn_time_', 'product_', 'yield_']
        data_lists = self.build_rxn_lists()
        for i in range(len(headers)):
            new_df = pd.DataFrame(data_lists[i])
            df_headers = self.create_column_headers(new_df, headers[i])
            new_df.columns = df_headers
            if i ==0:
                full_df = new_df
            else:
                full_df = pd.concat([full_df, new_df], axis=1)
        return full_df
    
        

    def main(self):
        # This function doesn't return anything. Instead, it saves the requested data as a pickle file at the path you see below
        # So you need to unpickle the data to see the output
        if 'uspto' in self.filename:
            full_df = self.build_full_df()
            #cleaned_df = self.clean_df(full_df)
            

            #save data to pickle
            filename = self.data.name
            full_df.to_pickle(f"data/USPTO/pickled_data/{filename}.pkl")
            
            #save names to pickle
            #list of the names used for molecules, as opposed to SMILES strings
            #save the names_list to pickle file
            with open(f"data/USPTO/molecule_names/molecules_{filename}.pkl", 'wb') as f:
                pickle.dump(self.names_list, f)
            
        #else:
            #print(f'The following does not contain USPTO data: {self.filename}')
            
        
def get_file_names():
    # Set the directory you want to look in
    directory = "data/USPTO/ord-data/data/"

    # Use listdir to get a list of all files in the directory
    folders = os.listdir(directory)
    files = []
    # Use a for loop to iterate over the files and print their names
    for folder in folders:
        if not folder.startswith("."):
            new_dir = directory+folder
            file_list = os.listdir(new_dir)
            # Check if the file name starts with a .
            for file in file_list:
                if not file.startswith("."):
                    new_file = new_dir+'/'+file
                    files += [new_file]
    return files

def merge_pickled_mol_names():
    #if the file already exists, delete it
    output_file_path = "data/USPTO/molecule_names/all_molecule_names.pkl"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    #create one big list of all the pickled names
    folder_path = 'data/USPTO/molecule_names/'
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_lst = []
    for file in tqdm(onlyfiles):
        if file[0] != '.': #We don't want to try to unpickle .DS_Store
            filepath = folder_path+file 
            unpickled_lst = pd.read_pickle(filepath)
            full_lst = full_lst + unpickled_lst
            
    unique_molecule_names = list(set(full_lst))
    
    #pickle the list
    with open(output_file_path, 'wb') as f:
        pickle.dump(unique_molecule_names, f)
    
def canonicalize_smiles(smiles):
    block = BlockLogs()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)
    
def build_solvents_set_and_dict():
    solvents = pd.read_csv('data/USPTO/solvents.csv', index_col=0)
    
    solvents['canonical_smiles'] = solvents['smiles'].apply(canonicalize_smiles)
    
    solvents_set = set(solvents['canonical_smiles'])
    
    
    # Combine the lists into a sequence of key-value pairs
    key_value_pairs = zip(list(solvents['stenutz_name']) + list(solvents['cosmo_name']), list(solvents['canonical_smiles']) + list(solvents['canonical_smiles']))

    # Create a dictionary from the sequence
    solvents_dict = dict(key_value_pairs)
    
    return solvents_set, solvents_dict

   
def build_replacements():
    block = BlockLogs()
    molecule_replacements = {}
     
    # Add a catalyst to the molecule_replacements dict (Done by Alexander)
    molecule_replacements['CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3].[Rh+3]'] = 'CC(=O)[O-].[Rh+2]'
    molecule_replacements['[CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3]]'] = 'CC(=O)[O-].[Rh+2]'
    molecule_replacements['[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]'] = 'CC(C)(C)[PH]([Pd][PH](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C'
    molecule_replacements['CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-]'] = 'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]'
    molecule_replacements['[CCO.CCO.CCO.CCO.[Ti]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    molecule_replacements['[CC[O-].CC[O-].CC[O-].CC[O-].[Ti+4]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    molecule_replacements['[Cl[Ni]Cl.c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1]'] = 'Cl[Ni]1(Cl)[P](c2ccccc2)(c2ccccc2)CCC[P]1(c1ccccc1)c1ccccc1'
    molecule_replacements['[Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1]'] = 'Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1'
    molecule_replacements['[Cl[Pd+2](Cl)(Cl)Cl.[Na+].[Na+]]'] = 'Cl[Pd]Cl'
    molecule_replacements['Karstedt catalyst'] =   'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    molecule_replacements["Karstedt's catalyst"] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    molecule_replacements['[O=C([O-])[O-].[Ag+2]]'] = 'O=C([O-])[O-].[Ag+]'
    molecule_replacements['[O=S(=O)([O-])[O-].[Ag+2]]'] = 'O=S(=O)([O-])[O-].[Ag+]'
    molecule_replacements['[O=[Ag-]]'] = 'O=[Ag]'
    molecule_replacements['[O=[Cu-]]'] = 'O=[Cu]'
    molecule_replacements['[Pd on-carbon]'] = '[C].[Pd]'
    molecule_replacements['[TEA]'] = 'OCCN(CCO)CCO'
    molecule_replacements['[Ti-superoxide]'] = 'O=[O-].[Ti]'
    molecule_replacements['[[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1]'] = '[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([P]([Pd][P](c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[sulfated tin oxide]'] = 'O=S(O[Sn])(O[Sn])O[Sn]'
    molecule_replacements['[tereakis(triphenylphosphine)palladium(0)]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['tetrakistriphenylphosphine palladium'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[zeolite]'] = 'O=[Al]O[Al]=O.O=[Si]=O'
    
    # Molecules found among the most common names in molecule_names
    molecule_replacements['TEA'] = 'OCCN(CCO)CCO'
    molecule_replacements['hexanes'] = 'CCCCCC'
    molecule_replacements['Hexanes'] = 'CCCCCC'
    molecule_replacements['hexanes ethyl acetate'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['EtOAc hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['EtOAc-hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['ethyl acetate hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['cuprous iodide'] = '[Cu]I'
    molecule_replacements['N,N-dimethylaminopyridine'] = 'n1ccc(N(C)C)cc1'
    molecule_replacements['dimethyl acetal'] = 'CN(C)C(OC)OC'
    molecule_replacements['cuprous chloride'] = 'Cl[Cu]'
    molecule_replacements["N,N'-carbonyldiimidazole"] = 'O=C(n1cncc1)n2ccnc2'
    # SiO2
    # Went down the list of molecule_names until frequency was 806

    # Iterate over the dictionary and canonicalize each SMILES string
    for key, value in molecule_replacements.items():
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            molecule_replacements[key] = Chem.MolToSmiles(mol)
        
        
    return molecule_replacements


def main(file, merge_cat_and_reag):
    
    manual_replacements_dict = build_replacements()
    solvents_set, solvents_dict = build_solvents_set_and_dict()
    replacements_dict = manual_replacements_dict.update(solvents_dict)
    
    
    instance = OrdToPickle(file, merge_cat_and_reag, replacements_dict, solvents_set)
    instance.main()
    
    

if __name__ == "__main__":
    
    start_time = datetime.now()
    
    args = sys.argv[1:]
    
    try:
        merge_cat_and_reag = args[0]
        if merge_cat_and_reag == 'True':
            merge_cat_and_reag = True
        elif merge_cat_and_reag == 'False':
            merge_cat_and_reag = False
        else:
            raise IndexError
    except IndexError:
        print('Please enter True or False for the first argument')
        print('Example: python USPTO_extraction.py True')
     
    
    pickled_data_path = 'data/USPTO/pickled_data'
    molecule_name_path = 'data/USPTO/molecule_names'

    if not os.path.exists(pickled_data_path):
        os.makedirs(pickled_data_path)
    if not os.path.exists(molecule_name_path):
        os.makedirs(molecule_name_path)
    
    files = get_file_names()
    
    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(files)
    Parallel(n_jobs=num_cores)(delayed(main)(i, merge_cat_and_reag) for i in inputs)
    

    # Create a list of all the unique molecule names
    merge_pickled_mol_names()
    
    
    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))

