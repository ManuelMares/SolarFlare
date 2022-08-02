from aenum import Enum

class DatasetPath(Enum):
    Labels = "Resources/Sampled_labels.pck" 
    Inputs = "Resources/Sampled_inputs.pck"

    LabelsReady = "Resources/LabelsReady.pck"
    InputsReady = "Resources/InputsReady.pck"

    #Each processed dataset must be saved as Labels_index / Inputs_index

def Add_File(name, dir):
    from aenum import extend_enum
    
    extend_enum(DatasetPath, name, dir)
    print (f"*File added \n   Name: {name} \n   Dir: {dir}")