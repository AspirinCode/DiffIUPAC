"""SMILES-based tokenization utilities.
"""

__all__ = ("PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "UNK_TOKEN", "SUFFIX",
           "SPECIAL_TOKENS", "PAD_TOKEN_ID", "BOS_TOKEN_ID", "EOS_TOKEN_ID",
           "UNK_TOKEN_ID", "SMILESBPETokenizer", "SMILESAlphabet")

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, models, normalizers, processors, trainers
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast
import torch
import os.path as pt

SUFFIX, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "", "<pad>", "<s>", "</s>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID = range(4)

iupac_list_non = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
 '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
  '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61',
   '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
    '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', ';', '.', '.0', "'"]

iupac_list = ['R', 'S', 'H', 'N', 'E', 'Z', 'aR', 'aS', 'bR', 'bS', 'cR', 'cS', 'dR', 'dS', 'aH', 'bH', 'cH', 'aE', 'aZ', 'a,', 'a-', 'b,', 'b-', 'c,', 'c-', 'd,', 'd-', 'a]', 'b]', 'c]', 'd]', 'e]', 'f]', 'g]', 'h]', 'i]', 'j]', 'k]', 'l]', 'm]', '-', 'yl', ',', ')', '(', ']', '[', 'meth', 'phenyl', 'di', 'an', 'eth', 'oxy', 'prop', 'e', 'amino', 'oxo', 'fluoro', 'cyclo', 'o', 'amide', 'tri', 'chloro', 'but', 'hydroxy', 'a', 'one', 'pyridin', 'hydro', 'benzo', 'acet', 'l', 'en', 'ol', 'amine', 'ylamin', 'oxa', 'oyl', 'carboxamide', 'benz', 'piperidin', 'thia', 'ate', 'sulf', 'bromo', 'ylidene', 'pyrimidin', 'tetra', 'ic_acid', 'penta', 'pyrrolidin', 'sulfonyl', 'hexa', 'hex', 'ane', 'pyrazol', 'phenoxy', 'carbonyl', 'thiophen', 'aza', 'piperazin', 'azo', 'carboxylate', 'imidazol', 'furan', 'nitro', 'carbam', 'anilino', 'pent', 'd', 'tert-', 'benzen', 'indol', 'sulfon', 'carboxylic_acid', 'diazo', 'az', 'ene', 'quinolin', 'naphthalen', 'morpholin', 'ium', 'cyano', 'bi', 'bis', 'hepta', 'pyrrol', 'spiro', 'r', 'ole', 'azin', 'hydrochloride', 'urea', 'yn', 'azido', 'carbamate', 'pyrrolo', 'it', 'imidazo', 'pyrazin', 'guanidin', 'thio', 'pyrazolo', 'iodo', 'imino', 'sulfam', 'carbon', 'olidin', 'epin', 'isoquinolin', 'deca', 'anilin', 'quinazolin', 'nitrile', 'hydrazin', 'epan', 'pyridazin', 'chromen', 'octa', 'octan', 'thieno', 'in', 'amido', 'hept', 'thiol', 'hydroiodide', 'imid', 'isoindol', 'nona', 'pyrido', 'inden', 'carbazol', 'ox', 'dodeca', 'etidin', 'oct', 'phenol', 'imidazolidin', 'sil', 'carboxy', 'imido', 'phosphor', 'purin', 'phospha', 'fluoren', 'carbox', 'indazol', 'undeca', 'furo', 'tetradeca', 'cyclopenta[a]phenanthren', 'form', 'quinoxalin', 'trideca', 'hexadeca', 'imine', 'sulfinyl', 'octadeca', 'carba', 'dec', 'adamant', 'chloride', 'sila', 'icos', 'ine', 'ide', 'naphthyridin', 'heptadeca', 'thione', 'anthracen', 'dodec', 'oxir', 'pyran', 'hydrogen', 'pentadeca', 'oxido', 'carbo', 'henicos', 'deuterio', 'docos', 'non', 'id', 'tert-butyl(dimethyl)silyl', 'carbamic_acid', 'pyrano', 'nonadeca', 'tris', 'but-2-eno', 'ic', 'at', 'phosphate', 'hydrazide', 'aceton', 'octadec', 'sulfo', 'thiomorpholin', 'pyrimido', 'oxamide', 'carbonimidoyl', 'oxet', 'inan', 'sodium', 'al', '(2+)', 'oxide', 'phthalazin', 'benzal', 'carbohydrazide', 'bora', 'benzhydr', 'tetracos', 'bor', 'hexadec', 'ioda', 'azonia', 'isocyano', 'acridin', 'hydroxylamin', 'formamide', 'phenanthren', 'ul', 'indeno', 'xanthen', 'nitroso', 'tetradec', 'phosphin', 'olan', 'peroxy', 'phosphono', 'tetr', 'pyrazolidin', 'dicarbon', 'olate', 'tricos', 'hexacos', 'indolo', 'indolizin', 'phosphon', 'undec', 'chromeno', 'pentacos', 'pyrazino', 'thi', 'hydrate', 'bromide', 'uid', 'boronic_acid', 'trityl', 'cen', 'sulfate', 'isochromen', 'octacos', 'isocyanato', 'acetal', 'azide', 'dimethylacetamide', 'tetrakis', 'iridin', 'nonadec', 'naphtho', 'heptadec', 'pyren', 'heptacos', 'carbamimidamido', 'sulfinam', 'oxid', 'iodide', 'etheno', 'disulfon', 'potassium', 'chrysen', 'yne', 'phosphino', 'carboximidoyl', 'quinolizin', 'tert-butyl(diphenyl)silyl', 'formamid', 'thiochromen', 'porphyrin', 'dicyan', 'triacont', 'pteridin', '(3+)', 'sulfin', 'ar', 'pentadec', 'io', 'phenothiazin', 'undecyl', 'oxal', 'phospho', 'borin', 'uide', 'uranium', 'picen', 'hydrobromide', 'cinnolin', 'isoindolo', 'phthal', 'phenac', 'phenanthridin', 'azino', 'tridec', 'zirconium', 'len', 'phenanthrolin', 'platinum', 'phenolate', 'sulfonato', 'oxybenzon', 'zinc', 'chlora', 'hydroperoxy', 'yttrium', 'pyrrolizin', 'carbothioyl', 'sel', 'iron', 'spirobi', 'copper', 'triphenylen', 'titanium', 'perox', 'nonacos', '(1+)', 'tridecyl', 'lithium', 'tetrol', '(4+)', 'carboxylato', 'thiopyran', 'pentacont', 'etan', 'iridium', 'thioxanthen', 'nickel', 'phenoxazin', 'hexatriacont', 'azulen', 'tetracont', 'tritriacont', 'azon', 'carbono', 'sulfino', 'dotriacont', 'stann', 'nitrate', 'broma', 'on', 'et', 'acetylen', 'fluoride', 'isothiocyanato', 'magnesium', 'cobalt', 'acenaphthylen', 'sulfamate', 'ruthenium', 'aldehyde', 'phosphite', 'nonafl', 'palladium', 'pentadecyl', 'purino', 'tetratriacont', 'epoxy', 'aluma', 'phenanthro', 'phenazin', 'fluoranthen', 'sulfinato', 'ocin', 'hentriacont', 'azanida', 'stanna', 'toluen', 'ylidyne', 'thiopyrano', 'perchlorate', 'calcium', 'mono', 'tungsten', 'sulfur', 'cyanamide', 'tricarbon', 'chlorid', 'dehydro', 'pyridazino', 'sulfido', 'irin', 'phosph', 'iran', 'thiocyanate', 'hypoiodite', 'ylium', 'imidazolo', 'octatriacont', 'dimethylurea', 'heptadecyl', 'tritio', 'hydrazono', 'selena', 'cyanide', 'dotetracont', 'isoquinolino', 'diazonium', 'pentatriacont', 'hydroxide', 'manganese', 'chromium', 'pentakis', 'hypofluorite', 'tin', 'sulfono', 'phosphoroso', 'vanadium', 'boranuida', 'ecin', 'hexakis', 's-indacen', 'os', 'fluoreno', 'mercury', 'sulfamic_acid', 'thiochromeno', 'phenalen', 'rhodium', 'amid', 'sulfite', 'ocan', 'phosphonato', 'heptatriacont', 'nonatriacont', 'borono', 'silver', 'gold', 'isothiochromen', 'nitron', 'hafnium', 'hexacont', '(2-)', 'hypochlorite', 'arsa', 'diphosphat', 'molybdenum', 'thallium', 'nonadecyl', 'fluora', 'nonatetracont', 'rhenium', 'tetracarbon', 'perylen', 'diphosphon', 'cyanate', 'oxygen', 'germ', 'nitramide', 'tell', 'aluminum', 'azuleno', 'quinolino', 'iod', 'actinium', 'terephthal', 'ecan', 'trithion', 'barium', 'hentetracont', 'dithion', 'phosphat', 'selenophen', 'xylen', 'germa', 'hen', 'perimidin', 'nitric_acid', 'rubidium', 'octatetracont', 'but-1-eno', 'nitramido', 'heptakis', 'thiocyanat', 'dibor', 'nitrous', 'hydrazon', 'thianthren', 'dili', 'hydride', 'oxonio', 'tetratetracont', 'isochromeno', 'dihydropter', 'indolizino', 'osmium', 'phosphonia', 'oxanthren', 'diazano', 'do', 'cyanato', 'diacetamid', 'oxam', 'silicate', 'cadmium', 'hydrofluoride', 'hexatetracont', 'boron', 'phosphindol', 'phenoxathiin', 'phosphonous_acid', 'octakis', 'bismuth', 'chromenylium', 'corrin', 'pyrylium', 'thion', 'cinnam', 'tritetracont', 'nitrite', 'gadolinium', 'diazonio', 'antimony', 'oxalo', 'onic_acid', 'biphenylen', 'sulfonio', 'cesium', 'oxonium', 'stiba', 'styren', 'heptacont', 'selenol', 'chloroform', 'diselen', 'onin', 'oxaldehyd', 'cerium', 'technetium', '(1-)', 'lead', 'ite', 'acenaphthyleno', 'dicarboximid', 'oxonia', 'strontium', '(5+)', 'iodid', 'lanthanum', 'rutherfordium', 'perchloric_acid', 'iren', 'tricosyl', 'hypobromite', 'europium', 'isocyanate', 'ido', 'iodosyl', 'nitrilium', 'neodymium', 'peroxide', 'pentatetracont', 'phenylen', 'tantalum', 'hect', 'buta-1,3-dieno', 'samarium', 'galla', 'methylal', 'fluorid', 'praseodymium', 'ytterbium', 'dimethoxyethane', 'scandium', 'seleno', 'dimethoxyethan', 'octacont', 'cub', 'gallium', 'diphosphate', 'pentacosyl', 'thalla', 'ous_acid', 'selenoate', 'arson', 'niobium', 'alumina', 'anisol', 'beryllium', 'thioph', 'heptatetracont', 'onan', 'tellura', 'quinoxalino', 'indiga', 'heptacosyl', 'isothiocyanate', 'inin', 'diphospho', 'thionia', 'selenido', 'nonacosyl', 'terbium', '(6+)', 'indig', 'dysprosium', 'quinazolino', 'iodyl', 'indium', 'hexatriacontyl', 'thiopyr', 'triphosphon', 'thorium', 'carbohydrazonoyl', 'as-indacen', 'fluoroform', 'erbium', 'phosphindolo', 'lutetium', 'selenopheno', 'arsin', 'arsor', 'iodat', 'silanuida', 'plumba', 'plumb', 'borano', 'sulfonium', 'tellurophen', 'indazolo', 'nitroxyl', 'nitrogen', 'anthra', 'isophosphindol', 'disulfid', 'nonacont', 'selone', 'iodonio', 'onate', 'trili', 'iodine', 'seleninyl', 'phenoxaphosphinin', 'phen', 'thulium', 'chloryl', 'phosphinimyl', 'cyanic_acid', 'acridophosphin', 'tetrali', 'cumen', 'holmium', 'selenopyran', 'dibenzamid', 'nitrous_acid', 'phthalal', 'selenocyanate', 'argon', 'iodate', 'isothiochromeno', 'mercurio', 'sulfide', 'bromid', 'iodonia', 'disulfate', 'fluorine', 'aceanthrylen', 'coronen', 'phenoxid', 'hydrazonic', 'telluro', 'silicon', 'chloronio', 'hypochlorous_acid', 'dodecakis', 'hydroseleno', 'phosphinolin', 'inda', 'phenaleno', 'phenylene', 'arsenic', 'chlorosyl', 'perchloryl', 'chlorate', 'bism', 'onat', 'terephthalal', '7,8-dihydropter', 'silano', 'boranthren', 'fermium', 'phosphano', 'arsoroso', 'hydrido', 'alum', 'selenium', 'pol', 'nonakis', 'stibo', 'phospheno', 'astatine', 'phosphanida', 'phenophosphazinin', 'stibor', 'sulfenat', 'silanida', 'pyranthren', 'arsono', 'decakis', 'oxaldehyde', 'cyanid', 'neptunium', 'diphosphor', 'bromate', 'selenate', 'selenin', 'selenonyl', 'phenoselenazin', 'hypoiodous_acid', 'silanylia', 'ditellur', 'arso', 'helicen', 'americium', 'pyreno', 'selenoxanthen', 'amoyl', 'telluroate', 'selen', 'selenochromen', 'diyl', 'dithianon', 'ose', 'plutonium', 'silicic_acid', '5,6,7,8-tetrahydropter', 'xenon', 'sulfamide', 'bisma', 'germanium', 'triphosphate', 'triphospho', 'triselen', 'isocyanide', 'isophosphinolin', 'tetrasulfide', 'dict', 'bromine', 'curium', 'acephenanthrylen', 'promethium', 'phosphanthridin', 'gall', 'selenocyanat', 'stilben', 'disulfide', 'isochromenylium', 'tetrathion', 'thall', 'selenat', 'chlor', 'silanthren', '(3-)', 'tetradecakis', 'xantheno', 'chromio', 'chlorite', 'californium', 'tetraphosphat', 'chlorine', 'iodoform', 'telluropyran', 'polona', 'lawrencium', 'naphthyridino', 'selenon', 'phenoxarsinin', 'as-indaceno', 'mercura', 'periodate', 'selenite', 'hypofluorous_acid', 'adip', 'bromyl', 'arsino', 'tungstenio', 'tellurochromen', 'stibin', 'trisulfide', 'isoselenochromen', 'zircona', 'hexali', 'tetraphosphate', 'onamide', 'chloronia', 'thiochromenylium', 'phosphorus', 'titana', 'dicyclohexylurea', 'phenarsazinin', '(8+)', 'nitroform', 'molybdenio', 'undecakis', 'rubicen', 'diselenid', 'triphosphat', 'diboron', 'trisulfid', 'hexadecakis', 'pleiaden', 'ter', 'arsonous_acid', 'ars', 'permangan', 'methoxychlor', 'tellurinyl', 'triacetamid', 'isocyanatid', '(7+)', 'phthalazino', 'chloric_acid', 'stibon', 'tellone', 'stib', 'protactinium', 'fluor', 'arsonato', 'einsteinium', 'tellur', 'molybda', 'telluroxanthen', 'water', 'pentali', 'vanadio', 'formazan', 'ovalen', 'brom', 'thioxantheno', 'selenomorpholin', 'arsonium', 'nobelium', 'cinnolino', 'nitrid', 'telluropyrano', 'neo', 'tellurate', 'bromic_acid', 'phosphinolino', 'iodite', 'arsindol', 'phosphen', 'tribenzamid', 'tellurium', 'oxyl', 'icosakis', 'tellurat', 'krypton', 'bromite', 'tridecakis', 'all', 'isotellurochromen', 'diarsor', 'bromosyl', 'helium', 'disulfite', 'deuteride', 'carboselenoyl', 'bromoform', 'trinaphthylen', 'octali', 'furano', 'selenino', 'iodic_acid', 'hydrotelluro', 'boronia', 'phosphinolizin', 'prism', 'periodic_acid', 'orot', 'pentadecakis', 'polonium', 'hexasulfide', 'stibono', 'selenanthren', 'ozone', 'phosphindolizin', 'urana', 'pyridino', 'phenotellurazin', 'meitnerium', 'tetrasulfid', 'selenonia', 'hypobromous_acid', 'selenopyrano', 'chlorat', 'trifluoromethanesulfonimid', 'seaborgium', 'azor', 'azonous_acid', 'selenoph', 'periodyl', 'perbromate', 'oson', 'berkelium', 'tungsta', 'ribo', 'pentaphosphate', 'hafna', 'telluropheno', 'tellurite', 'nitronium', 'mon', 'astata', 'isothiocyanatid', 'dubnium', 'isothiochromenylium', 'tellurin', 'sodio', 'selenono', 'selenochromeno', 'nitrosyl', 'mendelevium', 'ous', 'neon', 'fluoronio', 'azid', 'then', 'stannanylia', 'potassio', 'phosphanthren', 'disilic', 'chlorazin', 'titanio', 'bromat', 'triacontakis', 'pentasulfide', 'nonadecakis', 'rhenio', 'platina', 'phenoxatellurin', 'pentazocine', 'ferrio', 'cos', 'vanada', 'triselenid', 'telluronyl', 'tellurocyanate', 'pentazocin', 'fulven', 'distibor', 'diphosphite', 'radon', 'pentathion', 'nitrous_oxide', 'ferra', 'ditelluron', 'bis(trifluoromethylsulfonyl)imid', 'acridino', 'telluron', 'isophosphinolino', 'diselenon', 'diarson', 'stibanuida', 'germano', 'xanthylium', 'tert-butyl(dimethyl)silanyl', 'radium', 'osma', 'chlorous_acid', 'bromonio', 'arsonia', 'arsinolin', 'amate', 'urazol', 'triphosphor', 'nonali', 'deutero', 'nioba', 'acridarsin', 'yttrio', 'tert-butyl-dimethylsilyl', 'pyrimidino', 'pteridino', 'phenoxaselenin', 'isocyanid', 'irida', 'heptadecakis', 'bohrium', 'pentacosakis', 'octadecakis', 'thianthreno', 'telluroph', 't-', 'isophosphindolo', 'isoarsindol', 'henicosakis', '(4-)', 'ruthena', 'heptali', 'arsen', 'telluranthren', 'chryseno', 'carbotelluroyl', 'quinolizino', 'nonacosakis', 'francium', 'ethion', 'chroma', 'arsanthridin', 'arsanthren', 'tricosakis', 'tetraphosphor', 'tetracosakis', 'tellurocyanat', 'stibonia', 'stibonato', 'phosphanuida', 'phenoxathiino', 'manganio', 'eicosa', 'cobaltio', 'cera', 'amic_acid', 'stibino', 'stannanuida', 'samario', 's-indaceno', 'praseodymio', 'phenoxastibinin', 'pallada', 'neodymio', 'isoselenocyanate', 'germanuida', 'diazoamino', 'telluronia', 'tantalio', 'phenoxyl', 'phenothiarsinin', 'oxanthreno', 'octacosakis', 'mangana', 'lanthanio', 'isoarsinolin', 'indan', 'hexacosakis', 'hassium', 'arsinolizin', 'alli', 'thioxanth', 'tert-butyl(diphenyl)silanyl', 'stronta', 'stannano', 'rhodio', 'rhoda', 'praseodyma', 'phenazino', 'pentaphosphat', 'nitric', 'methoxyl', 'magnesio', 'dichrom', 'chlorazine', 'californa', 'butoxyl', 'bromous_acid', 'azonic_acid', 'arsinolino', 'arsindolo', 'arsindolizin', 'allo', 'actina', 'uronic_acid', 'thora', 'telluromorpholin', 'stibonium', 'stibano', 'rhena', 'phosphinolizino', 'phenothiazino', 'perbromyl', 'niobio', 'nickelio', 'isotellurochromeno', 'isoselenocyanato', 'iodous_acid', 'iodous', 'hydroselenonyl', 'dysprosio', 'cyclopenta[a]phenanthr', 'cerio', 'bara', 'aurio', 'arsanuida', 'ytterbio', 'uronate', 'tol', 'thulio', 'tert-butyl-diphenylsilyl', 'tellurono', 'stannanida', 'scandio', 'propoxyl', 'periodic', 'perbromic_acid', 'nitror', 'lutetio', 'isothiocyanic_acid', 'iridio', 'iodic', 'hypobor', 'hydroxyl', 'hydroseleninyl', 'holmio', 'hexasulfid', 'heptacosakis', 'gadolinio', 'europio', 'ethoxyl', 'erbio', 'docosakis', 'chlorous', 'chloric', 'arsinimyl', 'argentio', '▁', 'c', 'm', 't', 'p', 'n', 'u', 's', 'i', 'is', 'g', 'x', 'y', 'h', 'b', 'v', 'th', 'f', 'ph', 'hy', '▁p', 'cy', 'yc', 'im', 'ti', 'ch', 'ut', 'cys', 'st', '▁h', 'pi', 'uc', 'us', '▁b', '▁g', '▁c', 'ys', 'ct', '▁hy', 'gu', 'sp', 'xy', '▁s', 'yp', 'um', 'xim', 'thy', 'ps', 'fu', '▁cy', 'mph', '▁n', 'ni', '▁m', 'nth', 'cu', 'phth', 'ip', '▁f', 'ty', '▁cu', 'ym', 'ff', 'uf', 'fi', 'pt', 'tun', 'yt', '▁ch', '▁ps', '▁sty', '▁phyt', 'ub', 'mb', '▁fu', 'if', 'ci', '▁sym', 'ss', 'up', 'sty', '▁t', 'pp', 'mi', 'gn', 'ms', '▁pi', 'ist', 'tig', '▁thy', 'vii', 'hi', 'sym', '▁sub', 'ptu', 'cti', 'ig', 'tu', '▁fuc', '▁sy', '▁th', 'uv', 'si', '▁cys', 'bu', 'mu', 'vi', 'mp', 'ib', 'pu', '▁i', '▁bu', '▁gu', '▁mu', '▁st', 'un', 'uct', '▁u']

class SMILESBPETokenizer(BaseTokenizer):
    """Tokenizes SMILES strings and applies BPE.

    Args:
        vocab (`str` or `dict`, optional, defaults to `None`):
            Token vocabulary.
        merges (`str` or `dict` or `tuple`, optional, defaults to `None`):
            BPE merges.
        unk_token (`str` or `tokenizers.AddedToken`, optional, defaults to "<unk>")
        suffix (`str`, defaults to "")
        dropout (`float`, defaults to `None`)

    Examples:
        >>> tokenizer = SMILESBPETokenizer()
        >>> tokenizer.train("path-to-smiles-strings-file")
        Tokenization logs...
        >>> tokenizer.save_model("checkpoints-path")
        >>> same_tokenizer = SMILESBPETokenizer.from_file("checkpoints-path/vocab.json",
        ...                                               "checkpoints-path/merges.txt")
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        suffix: str = SUFFIX,
        dropout: Optional[float] = None,
    ) -> None:
        unk_token_str = str(unk_token)

        tokenizer = Tokenizer(models.BPE(vocab, merges, dropout=dropout,
                                         unk_token=unk_token_str,
                                         end_of_word_suffix=suffix))

        if tokenizer.token_to_id(unk_token_str) is not None:
            tokenizer.add_special_tokens([unk_token_str])

        tokenizer.normalizer = normalizers.Strip(left=False, right=True)
        tokenizer.decoder = decoders.Metaspace(add_prefix_space=True)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[(BOS_TOKEN, BOS_TOKEN_ID), (EOS_TOKEN, EOS_TOKEN_ID)])

        parameters = {"model": "BPE", "unk_token": unk_token, "suffix": suffix,
                      "dropout": dropout}

        super().__init__(tokenizer, parameters)

    @classmethod
    def from_file(cls, vocab_filename: str, merges_filename: str, **kwargs) \
            -> "SMILESBPETokenizer":
        vocab, merges = models.BPE.read_file(vocab_filename, merges_filename)
        return cls(vocab, merges, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Iterator,
        vocab_size: int = 1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

    @staticmethod
    def get_hf_tokenizer(
        tokenizer_file: str,
        special_tokens: Optional[Dict[str, str]] = None,
        model_max_length: int = 512,
        *init_inputs, **kwargs
    ) -> PreTrainedTokenizerFast:
        """Gets HuggingFace tokenizer from the pretrained `tokenizer_file`. Optionally,
        appends `special_tokens` to vocabulary and sets `model_max_length`.
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                            *init_inputs, **kwargs)
        special_tokens = special_tokens or dict(zip(
            ["pad_token", "bos_token", "eos_token", "unk_token"],
            SPECIAL_TOKENS))
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.model_max_length = model_max_length
        return tokenizer


@dataclass(init=True, eq=False, repr=True, frozen=True)
class SMILESAlphabet(Collection):
    atoms: FrozenSet[str] = frozenset([
        'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
        'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
        'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
        'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
        'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
        'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
        'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
        'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
        'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
        'Zn', 'Zr'
    ])

    # Bonds, charges, etc.
    non_atoms: FrozenSet[str] = frozenset([
        '-', '=', '#', ':', '(', ')', '.', '[', ']', '+', '-', '\\', '/', '*',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '@', 'AL', 'TH', 'SP', 'TB', 'OH',
    ])
    
    additional: FrozenSet[str] = frozenset()

    def __contains__(self, item: Any) -> bool:
        return item in self.atoms or item in self.non_atoms

    def __iter__(self):
        return (token for token in chain(self.atoms, self.non_atoms))

    def __len__(self) -> int:
        return len(self.atoms) + len(self.non_atoms) + len(self.additional)

    def get_alphabet(self) -> Set[str]:
        alphabet = set()
        for token in self.atoms:
            if len(token) > 1:
                alphabet.update(list(token))
                alphabet.add(token[0].lower())
            else:
                alphabet.add(token)
                alphabet.add(token.lower())
        for token in chain(self.non_atoms, self.additional):
            if len(token) > 1:
                alphabet.update(list(token))
            else:
                alphabet.add(token)
        return alphabet

def get_smiles_tokenizer(is_train=1,checkpoint = "./data/smile_tocken"):

    tokenizer_filename = f"{checkpoint}/tokenizer.json"
    #filename = "./data/smile_tocken/train_data_new.csv"

    filename = "./data/smile_tocken/train_data.csv"


    hyperparams = {"batch_size": 256, "max_epochs": 10, "max_length": 500,
               "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 1_000, "final_learning_rate": 5e-8,
               "vocab_size": 200, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 8, "n_head": 8, "n_embd": 256}

    alphabet = list(SMILESAlphabet().get_alphabet())
    tokenizer = SMILESBPETokenizer(dropout=None)
    tokenizer.train(filename,
                vocab_size=hyperparams["vocab_size"] + len(alphabet),
                min_frequency=hyperparams["min_frequency"],
                initial_alphabet=alphabet)
    tokenizer.save_model(checkpoint)
    tokenizer.save(tokenizer_filename)

    tokenizer = SMILESBPETokenizer.get_hf_tokenizer(tokenizer_filename, model_max_length=hyperparams["max_length"])

    if is_train:
        torch.save(tokenizer, pt.join(checkpoint,"real_smiles_tokenizer.pt"))
        print("smiles_tokenizer saving...",len(tokenizer))
    else:
        tokenizer = torch.load(pt.join(checkpoint,"real_smiles_tokenizer.pt"), map_location="cpu")
        print("smiles_tokenizer loading...",len(tokenizer))

    return tokenizer

if __name__ == "__main__":

    tokenizer = get_smiles_tokenizer(is_train=1,checkpoint = "./data/smile_tocken")
    
    smiles_string = "CC(Cl)=CCCC=C(C)Cl"
    smiles_encoded = tokenizer(smiles_string)
    smiles_merges = tokenizer.convert_ids_to_tokens(smiles_encoded["input_ids"])

    print(smiles_encoded)
    print(smiles_merges)


    line_number = 1

    valid_line=[]

    with open("data/pubchem_smiles.csv",'r') as f:
        myline = f.readline()
        while myline:
            #print("line_number:",line_number)

            iupac_encoded = tokenizer(myline)
            iupac_merges = tokenizer.convert_ids_to_tokens(iupac_encoded["input_ids"])
            #print(iupac_encoded)
            #print(iupac_merges)

            if iupac_encoded["input_ids"].count(1)==1:
                valid_line.append(myline)

            if line_number%50000==0:
                with open("data/pubchem_smiles_valid.csv",'a') as ff:
                    for j in valid_line:
                        ff.write(j)
                valid_line=[]

            myline = f.readline()
            line_number = 1+line_number
    