import streamlit as st
import re

# Mapping NER tags to colors
NER_COLORS = {
    "REGULATION_REF": "#FFD700",     # gold
    "ACTOR_ROLE": "#87CEFA",         # light blue
    "AUTHORITY": "#90EE90",          # light green
    "DATA_CATEGORY": "#FFB6C1",      # light pink
    "PROCESSING_OPERATION": "#FFA500" # orange
}

# Sample text with inline XML tags (replace with your full text or load from file)
TEXT = """
<CHAPTER II>
<REGULATION_REF>Article 5</REGULATION_REF>
Prohibited AI practices
1. The following AI practices shall be prohibited:
(a) the placing on the market, the putting into service or the use of an AI system that deploys subliminal techniques beyond a person’s consciousness or purposefully manipulative or deceptive techniques, with the objective, or the effect of materially distorting the behaviour of a person or a group of persons by appreciably impairing their ability to make an informed decision, thereby causing them to take a decision that they would not have otherwise taken in a manner that causes or is reasonably likely to cause that person, another person or group of persons significant harm;
(b) the placing on the market, the putting into service or the use of an AI system that exploits any of the vulnerabilities of a natural person or a specific group of persons due to their age, disability or a specific social or economic situation, with the objective, or the effect, of materially distorting the behaviour of that person or a person belonging to that group in a manner that causes or is reasonably likely to cause that person or another person significant harm;
(c) the placing on the market, the putting into service or the use of AI systems for the evaluation or classification of natural persons or groups of persons over a certain period of time based on their social behaviour or known, inferred or predicted <DATA_CATEGORY>personal</DATA_CATEGORY> or personality characteristics, with the social score leading to either or both of the following:
(i) detrimental or unfavourable treatment of certain natural persons or groups of persons in social contexts that are unrelated to the contexts in which the data was originally generated or collected;
(ii) detrimental or unfavourable treatment of certain natural persons or groups of persons that is unjustified or disproportionate to their social behaviour or its gravity;
(d) the placing on the market, the putting into service for this specific purpose, or the use of an AI system for making risk assessments of natural persons in order to assess or predict the risk of a natural person committing a criminal offence, based solely on the <PROCESSING_OPERATION>profiling</PROCESSING_OPERATION> of a natural person or on assessing their personality traits and characteristics; this prohibition shall not apply to AI systems used to support the human assessment of the involvement of a person in a criminal activity, which is already based on objective and verifiable facts directly linked to a criminal activity;
(e) the placing on the market, the putting into service for this specific purpose, or the use of AI systems that create or expand <DATA_CATEGORY>facial recognition</DATA_CATEGORY> databases through the untargeted scraping of <DATA_CATEGORY>facial images</DATA_CATEGORY> from the internet or CCTV footage;
(f) the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons;
12.7.2024 objectives specified in paragraph 1, first subparagraph, point (h), as identified in the request and, in particular, remains limited to what is strictly necessary concerning the period of time as well as the geographic and personal scope.
In deciding on the request, that <AUTHORITY>authority</AUTHORITY> shall take into account the elements referred to in paragraph 2. No decision that produces an adverse legal effect on a person may be taken based solely on the output of the ‘real-time’ remote biometric identification system.
(g) the placing on the market, the putting into service for this specific purpose, or the use of <DATA_CATEGORY>biometric categorisation systems</DATA_CATEGORY> that categorise individually natural persons based on their <DATA_CATEGORY>biometric data</DATA_CATEGORY> to deduce or infer their race, political opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation; this prohibition does not cover any labelling or filtering of lawfully acquired <DATA_CATEGORY>biometric datasets</DATA_CATEGORY>, such as images, based on <DATA_CATEGORY>biometric data</DATA_CATEGORY> or categorizing of <DATA_CATEGORY>biometric data</DATA_CATEGORY> in the area of law enforcement;
(h) the use of ‘real-time’ remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for the purposes of law enforcement, unless and in so far as such use is strictly necessary for one of the following objectives:
(i) the targeted search for specific victims of abduction, trafficking in human beings or sexual exploitation of human beings, as well as the search for missing persons;
(ii) the prevention of a specific, substantial and imminent threat to the life or physical safety of natural persons or a genuine and present or genuine and foreseeable threat of a terrorist attack;
(iii) the localisation or identification of a person suspected of having committed a criminal offence, for the purpose of conducting a criminal investigation or prosecution or executing a criminal penalty for offences referred to in <REGULATION_REF>Annex II</REGULATION_REF> and punishable in the Member State concerned by a custodial sentence or a detention order for a maximum period of at least four years.
Point (h) of the first subparagraph is without prejudice to <REGULATION_REF>Article 9</REGULATION_REF> of <REGULATION_REF>Regulation (EU) 2016/679</REGULATION_REF> for the processing of <DATA_CATEGORY>biometric data</DATA_CATEGORY> for purposes other than law enforcement.
2. The use of ‘real-time’ remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for the purposes of law enforcement for any of the objectives referred to in paragraph 1, first subparagraph, point (h), shall be deployed for the purposes set out in that point only to confirm the identity of the specifically targeted individual, and it shall take into account the following elements:
(a) the nature of the situation giving rise to the possible use, in particular the seriousness, probability and scale of the harm that would be caused if the system were not used;
(b) the consequences of the use of the system for the rights and freedoms of all persons concerned, in particular the seriousness, probability and scale of those consequences.
In addition, the use of ‘real-time’ remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for the purposes of law enforcement for any of the objectives referred to in paragraph 1, first subparagraph, point (h), of this Article shall comply with necessary and proportionate safeguards and conditions in relation to the use in accordance with the national law authorising the use thereof, in particular as regards the temporal, geographic and personal limitations.
The use of the ‘real-time’ remote <DATA_CATEGORY>biometric identification system</DATA_CATEGORY> in publicly accessible spaces shall be authorised only if the <AUTHORITY>law enforcement authority</AUTHORITY> has completed a fundamental rights impact assessment as provided for in <REGULATION_REF>Article 27</REGULATION_REF> and has registered the system in the EU database according to <REGULATION_REF>Article 49</REGULATION_REF>.
However, in duly justified cases of urgency, the use of such systems may be commenced without the registration in the EU database, provided that such registration is completed without undue delay.
3. For the purposes of paragraph 1, first subparagraph, point (h) and paragraph 2, each use for the purposes of law enforcement of a ‘real-time’ remote <DATA_CATEGORY>biometric identification system</DATA_CATEGORY> in publicly accessible spaces shall be subject to a prior authorisation granted by a <AUTHORITY>judicial authority</AUTHORITY> or an <AUTHORITY>independent administrative authority</AUTHORITY> whose decision is binding of the Member State in which the use is to take place, issued upon a reasoned request and in accordance with the detailed rules of national law referred to in paragraph 5.
However, in a duly justified situation of urgency, the use of such system may be commenced without an authorisation provided that such authorisation is requested without undue delay, at the latest within 24 hours.
If such authorisation is rejected, the use shall be stopped with immediate effect and all the data, as well as the results and outputs of that use shall be immediately discarded and deleted.
The competent <AUTHORITY>judicial authority</AUTHORITY> or an <AUTHORITY>independent administrative authority</AUTHORITY> whose decision is binding shall grant the authorisation only where it is satisfied, on the basis of objective evidence or clear indications presented to it, that the use of the ‘real-time’ remote <DATA_CATEGORY>biometric identification system</DATA_CATEGORY> concerned is necessary for, and proportionate to, achieving one of the 12.7.2024 4.
Without prejudice to paragraph 3, each use of a ‘real-time’ remote <DATA_CATEGORY>biometric identification system</DATA_CATEGORY> in publicly accessible spaces for law enforcement purposes shall be notified to the relevant <AUTHORITY>market surveillance authority</AUTHORITY> and the national <AUTHORITY>data protection authority</AUTHORITY> in accordance with the national rules referred to in paragraph 5.
The notification shall, as a minimum, contain the information specified under paragraph 6 and shall not include sensitive operational data.
5. A Member State may decide to provide for the possibility to fully or partially authorise the use of ‘real-time’ remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for the purposes of law enforcement within the limits and under the conditions listed in paragraph 1, first subparagraph, point (h), and paragraphs 2 and 3.
Member States concerned shall lay down in their national law the necessary detailed rules for the request, issuance and exercise of, as well as supervision and reporting relating to, the authorisations referred to in paragraph 3.
Those rules shall also specify in respect of which of the objectives listed in paragraph 1, first subparagraph, point (h), including which of the criminal offences referred to in point (h)(iii) thereof, the competent authorities may be authorised to use those systems for the purposes of law enforcement.
Member States shall notify those rules to the <AUTHORITY>Commission</AUTHORITY> at the latest 30 days following the adoption thereof.
Member States may introduce, in accordance with Union law, more restrictive laws on the use of remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY>.
6. National <AUTHORITY>market surveillance authorities</AUTHORITY> and the national <AUTHORITY>data protection authorities</AUTHORITY> of Member States that have been notified of the use of ‘real-time’ remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for law enforcement purposes pursuant to paragraph 4 shall submit to the <AUTHORITY>Commission</AUTHORITY> annual reports on such use.
For that purpose, the <AUTHORITY>Commission</AUTHORITY> shall provide Member States and national <AUTHORITY>market surveillance</AUTHORITY> and <AUTHORITY>data protection authorities</AUTHORITY> with a template, including information on the number of the decisions taken by competent <AUTHORITY>judicial authorities</AUTHORITY> or an <AUTHORITY>independent administrative authority</AUTHORITY> whose decision is binding upon requests for authorisations in accordance with paragraph 3 and their result.
7. The <AUTHORITY>Commission</AUTHORITY> shall publish annual reports on the use of real-time remote <DATA_CATEGORY>biometric identification systems</DATA_CATEGORY> in publicly accessible spaces for law enforcement purposes, based on aggregated data in Member States on the basis of the annual reports referred to in paragraph 6.
Those annual reports shall not include sensitive operational data of the related law enforcement activities.
8. This Article shall not affect the prohibitions that apply where an AI practice infringes other Union law.
"""

# Function to convert XML tags to HTML spans with colors
def annotate_text(text):
    def replace_tag(match):
        tag = match.group(1)
        content = match.group(2)
        color = NER_COLORS.get(tag, "#FFFFFF")
        return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px;">{content} <strong>[{tag}]</strong></span>'

    # Regex to match <TAG>content</TAG>
    annotated = re.sub(r"<([A-Z_]+)>(.*?)</\1>", replace_tag, text)
    return annotated

# Streamlit UI
st.title("Legal NER Annotator Viewer")
st.markdown("This app visualizes NER entities in legal text with colored tags.")

# Show annotated text
st.markdown(annotate_text(TEXT), unsafe_allow_html=True)
