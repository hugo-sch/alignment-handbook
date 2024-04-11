from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import torch

ARTICLE_TEXT_EXAMPLE = """[TEXT] I en sjokkerende avsløring har nylige etterforskninger brakt for dagen en omfattende svindelsak hvor milliarder av kroner, øremerket offentlige tjenester, er blitt ulovlig omdirigert inn i lommene på korrupte tjenestemenn og kvinner. Denne oppsiktsvekkende utviklingen har vekket stor offentlig forargelse og kaster en mørk skygge over det offentlige tillitsforholdet. Ifølge rapporter fra Økokrim, begynte etterforskningen etter at det ble oppdaget uvanlige transaksjoner i flere departementers økonomisystemer. Detaljerte granskninger avslørte et komplekst nettverk av fiktive kontrakter, overfakturering og direkte tyveri, som strekker seg over flere år. "Vi står overfor en av de mest sofistikerte og omfattende svindeloperasjonene i norsk historie," uttalte Økokrim-sjefen under en pressekonferanse. "Pengene som var ment for å styrke offentlige tjenester som helsevesen, utdanning og infrastruktur, har i stedet blitt brukt til å berike en liten gruppe individer." Det er avdekket at svindelen involverte flere nivåer av regjeringsansatte, fra lavere nivå administrativt personell til høyere tjenestemenn med tilgang til omfattende fond. Disse aktørene har angivelig samarbeidet om å omgå de økonomiske kontrollene som er på plass for å beskytte offentlige midler. Reaksjonene fra offentligheten har vært av både sjokk og vrede. Mange krever øyeblikkelig handling og strenge straffer for de involverte, samt omfattende reformer for å gjenopprette integriteten i offentlig forvaltning. Regjeringen har svart på skandalen med løfter om gjennomsiktighet og reform. Statsministeren har annonsert en uavhengig undersøkelse for å få full oversikt over omfanget av svindelen, samt tiltak for å styrke det økonomiske tilsynet. "Vi vil ta alle nødvendige skritt for å gjenopprette det norske folkets tillit. Det er uakseptabelt at midler som er ment å tjene offentligheten, blir stjålet av de som er betrodd å forvalte dem," sa statsministeren i en offisiell uttalelse. Detaljene i etterforskningen er fortsatt under utvikling, og myndighetene har lovet å holde offentligheten løpende informert. Dette er en sak som vil fortsette å dominere nyhetsoverskriftene i Norge i tiden som kommer. [TITLE] """

def load_model_and_adapters(adapter_path, base_model=None):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    if base_model is None:
        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                )
        base_model_path = peft_config.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=quantization_config)
    
    lora_model = PeftModel.from_pretrained(base_model, adapter_path, revision=peft_config.revision)
    lora_model = lora_model.merge_and_unload()
    
    return lora_model

def double_load_print():
    sft_model = load_model_and_adapters("data/ap-gpt-j-6b-sft-qlora-04-08")
    dpo_model = load_model_and_adapters("data/ap-gpt-j-6b-dpo-qlora-04-08", sft_model)
    tokenizer = AutoTokenizer.from_pretrained("data/ap-gpt-j-6b-dpo-qlora-04-10")
    
    generate_output(ARTICLE_TEXT_EXAMPLE, dpo_model, tokenizer, 50)

def generate_output(input_prompt, model, tokenizer, max_output_length=25):
    input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
    
    with torch.cuda.amp.autocast():
        generation_output = model.generate(
            input_ids=input_tokens,
            min_new_tokens=2,
            max_new_tokens=max_output_length,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=1.15,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    tokenizer_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print(tokenizer_output)
    
double_load_print()