from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

input_text = """[TEXT] I en sjokkerende avsløring har nylige etterforskninger brakt for dagen en omfattende svindelsak hvor milliarder av kroner, øremerket offentlige tjenester, er blitt ulovlig omdirigert inn i lommene på korrupte tjenestemenn og kvinner. Denne oppsiktsvekkende utviklingen har vekket stor offentlig forargelse og kaster en mørk skygge over det offentlige tillitsforholdet.

Ifølge rapporter fra Økokrim, begynte etterforskningen etter at det ble oppdaget uvanlige transaksjoner i flere departementers økonomisystemer. Detaljerte granskninger avslørte et komplekst nettverk av fiktive kontrakter, overfakturering og direkte tyveri, som strekker seg over flere år.

"Vi står overfor en av de mest sofistikerte og omfattende svindeloperasjonene i norsk historie," uttalte Økokrim-sjefen under en pressekonferanse. "Pengene som var ment for å styrke offentlige tjenester som helsevesen, utdanning og infrastruktur, har i stedet blitt brukt til å berike en liten gruppe individer."

Det er avdekket at svindelen involverte flere nivåer av regjeringsansatte, fra lavere nivå administrativt personell til høyere tjenestemenn med tilgang til omfattende fond. Disse aktørene har angivelig samarbeidet om å omgå de økonomiske kontrollene som er på plass for å beskytte offentlige midler.

Reaksjonene fra offentligheten har vært av både sjokk og vrede. Mange krever øyeblikkelig handling og strenge straffer for de involverte, samt omfattende reformer for å gjenopprette integriteten i offentlig forvaltning.

Regjeringen har svart på skandalen med løfter om gjennomsiktighet og reform. Statsministeren har annonsert en uavhengig undersøkelse for å få full oversikt over omfanget av svindelen, samt tiltak for å styrke det økonomiske tilsynet.

"Vi vil ta alle nødvendige skritt for å gjenopprette det norske folkets tillit. Det er uakseptabelt at midler som er ment å tjene offentligheten, blir stjålet av de som er betrodd å forvalte dem," sa statsministeren i en offisiell uttalelse.

Detaljene i etterforskningen er fortsatt under utvikling, og myndighetene har lovet å holde offentligheten løpende informert. Dette er en sak som vil fortsette å dominere nyhetsoverskriftene i Norge i tiden som kommer. [TITLE]"""

base_model = AutoModelForCausalLM.from_pretrained("sft_model")
peft_config = PeftConfig.from_pretrained("data/nb-gpt-j-6B-v2-dpo-qlora-true1")
lora_model = PeftModel.from_pretrained(base_model, "data/nb-gpt-j-6B-v2-dpo-qlora-true1", revision=peft_config.revision)
base_model = lora_model

tokenizer = AutoTokenizer.from_pretrained("sft_model")
tokenizer.padding_side = "left"

input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = base_model.generate(input_ids, max_new_tokens=50, temperature=0.7, do_sample=True)
output_text = tokenizer.decode(output[0])

print(output_text)