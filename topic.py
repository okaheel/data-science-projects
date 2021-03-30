# topic modeler that builds a small lda model based on input text document

doc1 = "WE-CAN (Western Wildfire Experiment for Cloud Chemistry, Aerosol, Absorption and Nitrogen) seeks to understand the chemistry in western wildfire smoke has major ramifications for air quality, nutrient cycles, weather and climate. This project will systematically characterize the emissions and first day of evolution of western U.S. wildfire plumes. It will focus on three sets of scientific questions related to fixed nitrogen, absorbing aerosols, cloud activation and chemistry in wildfire plumes. The data will be collected from the NCAR/NSF C-130 research aircraft, in coordination with planned NOAA and NASA field intensives."
doc2 = "For the 2018 field season (8 March to 15 April 2018) VORTEX-SE has a primary domain in the area of northern Alabama with two subdomains, the Sand Mountain domain over northeastern Alabama and the Western Domain that covers northern Alabama to the west of Huntsville.  A number of fixed and mobile facilities from the 2017 field season will also be available for 2018 including the University of Alabama-Huntsville (UAH) Mobile Alabama X-band (MAX) radar, UAH Mobile Integrated Profiling System (MIPS), UAH Rapidly Deployable Atmospheric Profiling System (RaDAPS), UAH Mobile Doppler Lidar and Sounding system (MoDLS), the University of Louisiana-Monroe (ULM)  mobile radiosonde systems, as well as boundary layer site at the SWIRLL building at UAH.  Additional instrumentation for the 2018 field season includes the NOAA P-3 aircraft and University of Oklahoma SMART-R mobile radars."
doc3 = "The IDEAL (Instabilities, Dynamics and Energetics accompanying Atmospheric Layering) field project combines ground-based in-situ observations with modeling efforts and will quantify the understanding of “sheet-and-layer” (S&L) structures, morphologies, and underlying dynamics in the stably stratified lower troposphere. The IDEAL project science goals are: Identify the multi-scale GW dynamics that appear contribute to local instabilities, intermittent turbulence, and the creation and evolution of S&L structures from ~50 m to ~2-3 km, Identify the instability dynamics [i.e. Kelvin-Helmholtz Instability (KHI), GW breaking,intrusions, others?] that drive turbulence and transport, and their scales, amplitudes, and heat and momentum fluxes, for various S&L scales and energy sources, and Quantify the scales, intensities, character, and consequences of turbulence events for various S&L and instability flows and strengths."
doc4 = "The Airborne Research Instrumentation Testing Opportunity 2017 (ARISTO2017) is a newly-created NSF-sponsored flight test program that will be conducted annually on one of the NSF/NCAR aircraft. The purpose of the ARISTO program is to provide regular flight test opportunities for newly developed or highly modified instruments as part of their development effort. The program was created in response to a critical need, expressed by the NSF community, for regularly scheduled flight-testing programs to be able to not only test instrumentation, but also data systems, inlets and software well ahead of a field campaign in order to maintain cutting-edge and vibrant airborne research."
doc5 = "The overarching goal of SNOWIE is to understand the natural dynamical and microphysical processes by which precipitation forms and evolves within orographic winter storms and to determine the physical processes by which cloud seeding with silver iodide (AgI), either from ground generators or aircraft, impacts the amount and spatial distribution of snow falling across a river basin. The core scientific objectives build on results of recent investigations of orographic clouds by the NSF-funded 2012-13 AgI Seeding Cloud Impact Investigation (ASCII) field program in Wyoming (Geerts et al. 2013; Pokharel et al. 2015). SNOWIE will be conducted in the Payette Mountains in Idaho and partners with Idaho Power Company (IPC) who maintains an operational seeding program in the region. SNOWIE uses similar observational and modeling tools as ASCII. However, unlike ASCII, which used only ground-based seeding, SNOWIE focuses a significant effort investigating cases of airborne seeding. This new focus allows detailed examination of the cloud microphysical response to AgI seeding in a region above the planetary boundary layer (PBL) that is accessible by the cloud-physics research aircraft and in turn provides a data set to evaluate and improve the numerical model’s ability to capture key details of ice and precipitation development in the studied clouds. SNOWIE differentiates itself from ASCII in other significant ways: (1) the seeding aircraft will, on a few flights, target a relatively simple stratus cloud away from the mountains, in order to repeat with state-of-the-art instruments the original experiment by Schaeffer (1946) and Vonnegut (1947) to allow for direct, unambiguous microphysical change-of-events verification; (2) natural and seeded storm structures will be analyzed in more detail, with higher temporal resolution, and over a larger domain, especially by means of one airborne and two scanning Doppler radars, in order to better isolate seeding signatures from natural cloud evolution; (3) aerosol size distributions and concentrations will be characterized using ground-based measurements in the airmass impinging on the target mountain; and (4) the modeling component of SNOWIE will also use historical data to evaluate seeding effectiveness. ASCII has shown to some degree how high-resolution cloud and aerosol resolving numerical simulations can be validated with observations and be used for seeding evaluation—but SNOWIE provides this and the ability to run the model retrospectively with and without seeding to quantify the fraction of seedable storms and the impact on the seasonal snowpack.."

doc_complete = [doc1, doc2, doc3, doc4, doc5]

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
doc_clean = [clean(doc).split() for doc in doc_complete]

import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics = 3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=4))