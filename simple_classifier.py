"""
Classificador simplificado sem fine-tuning (mais rápido para demonstração)
Usa análise de palavras-chave e padrões de texto
"""
import re

class ToxicityClassifier:
    def __init__(self):
        """Inicializa o classificador"""
        print("Carregando classificador...")
        
        # Lista de palavras e padrões tóxicos
        self.toxic_patterns = self._load_toxic_patterns()
        
        print("[OK] Classificador carregado!")
    
    def _load_toxic_patterns(self):
        """Carrega padrões de toxicidade"""
        return {
            'insultos': [
                # Insultos comuns
                r'\b(idiota|burr[oa]|estúpid[oa]|imbecil|otári[oa]|babac[oa])\b',
                r'\b(desgra[çc]ad[oa]|desgraç[oa]|miserável|verme|escroto)\b',
                r'\b(lixo|vagabund[oa]|safad[oa]|nojento|immundo)\b',
                r'\b(cret[ií]n[oa]|ment[ei]ros[oa]|palha[çc]o|ridícul[oa])\b',
                r'\b(in[úu]til|incompetente|fracassad[oa]|medíocre)\b',
                r'\b(retardad[oa]|mongolóide|débil mental|deficiente)\b',
                r'\b(filho da m[ãa]e|filha da m[ãa]e|sem vergonha)\b',
                r'\b(canalha|sacana|vigarista|malandro|bandid[oa])\b',
                r'\b(porco|porcaria|sujo|fedorento|seboso)\b',
                r'\b(trouxa|bobo|tonto|anta|jumento|asno)\b',
                r'\b(ralo|ralé|gentalha|escória|escumalha)\b',
                r'\b(peste|praga|demônio|capeta|diabo)\b',
                r'\b(bost[ao]|merda seca|bosta humana)\b',
                r'\b(cu|c[uú]zao|c[uú]zona|bundão|bundona)\b',
                r'\b(arrombad[oa]|fudid[oa]|ferrad[oa])\b'
            ],
            'palavrões': [
                # Palavrões pesados
                r'\b(caralho|cacete|porra|merda|bosta)\b',
                r'\b(puto|puta|put[oa]|putaria|puteiro)\b',
                r'\b(fdp|filho da puta|filha da puta|fda puta)\b',
                r'\b(fod[ea]|foder|fodido|vai se fod[ea]r)\b',
                r'\b(corno|corna|chifr[uda]|chifrudo)\b',
                r'\b(caralh[oa]|pqp|puta que pariu)\b',
                r'\b(buceta|xoxota|xana|racha|pepeca)\b',
                r'\b(pau|pinto|p[ií]ca|rola|c[aá]ralho)\b',
                r'\b(cu|[aâ]nus|bunda|rabo)\b',
                r'\b(puta merda|cacete|caraio|krl|krlo)\b',
                r'\b(vsf|vai tomar no [cú]|vai se ferrar)\b',
                r'\b(pqp|ptqp|poha|porra nenhuma)\b'
            ],
            'homofobia': [
                r'\b(viado|viada|bicha|baitola|maricas)\b',
                r'\b(fresco|frutinha|boiola|gay de araque)\b',
                r'\b(gay|lésbica|travesti|trans).*(lixo|nojo|imundo|anormal)\b',
                r'\b(sapat[ãa]o|machorra|caminhoneira)\b',
                r'\b(paneleiro|veado|invertido)\b'
            ],
            'racismo': [
                # Termos racistas diretos (animalizações)
                r'\b(macac[oa]|chimpanz[eé]|chipanz[eé]|gorila|s[ií]mio|primata)\b',
                r'\b(preto|negro|negra|pardo).*(sujo|fedido|macaco|chimpanzé)\b',
                r'\b(preto|negro).*(ladrão|bandido|vagabundo)\b',
                r'\b(crioulo|criola|nego|nega).*(safado|vagabundo)\b',
                r'\b(volta.*(África|senzala|tronco))\b',
                # Contextos racistas com "você"/"tu"/"seu"
                r'\b(você|tu|vc|voce|v[cç]).*(é|e|eh).*(macaco|chimpanz|chipanz|gorila|símio|primata)\b',
                r'\b(seu|sua|teu|tua).*(macaco|chimpanz|chipanz|gorila|símio|primata)\b',
                r'\b(parece|igual|tipo|nem parece).*(macaco|chimpanz|gorila|símio)\b',
                # Variações sem acento
                r'\b(voce|vc).*(e|eh).*(macaco|gorila|simio)\b',
                # Termos racistas adicionais
                r'\b(escravo|escrava|senzala|chibata)\b',
                r'\b(cor de (bosta|cocô|carvão))\b'
            ],
            'misoginia': [
                r'\b(mulher|feminista).*(burra|idiota|vadia|piranha)\b',
                r'\b(puta|vadia|piranha|galinha|vagabunda|safada)\b',
                r'\b(rapariga|quenga|prostituta|rameira)\b',
                r'\b(vagabunda|sem vergonha|oferecida|cachorra)\b',
                r'\b(mulher.*(lugar|cozinha|tanque|fogão))\b',
                r'\b(vaca|égua|cadela|cabra)\b'
            ],
            'ameaças': [
                r'\b(vou te (matar|socar|bater|esfaquear|quebrar|acabar))\b',
                r'\b(vai morrer|vou te pegar|vai se arrepender|vai se foder)\b',
                r'\b(te quebro|te arrebento|te acabo|te destruo)\b',
                r'\b(toma cuidado|fica esperto|você vai ver|vai ter problema)\b',
                r'\b(vou mandar|vou chamar|vou denunciar).*(mat[aá]r|bater)\b'
            ],
            'ofensas_gerais': [
                r'\b(vai.*(inferno|morrer|explodir|desgraça))\b',
                r'\b(toma no.*(cu|[aâ]nus|rabo|bunda))\b',
                r'\b(enfia no.*(cu|[aâ]nus|rabo))\b',
                r'\b(cala.*(boca|matraca|a boca))\b',
                r'\b(se mata|vai morrer|morre logo)\b',
                r'\b(nojento|asqueroso|repugnante|abominável)\b',
                r'\b(come merda|chupa rola|mama aqui)\b'
            ]
        }
    
    def classify(self, text):
        """
        Classifica um texto como tóxico ou não-tóxico
        
        Args:
            text (str): Texto a ser classificado
            
        Returns:
            dict: {"label": "TÓXICA" ou "NÃO TÓXICA", "confidence": float}
        """
        if not text or len(text.strip()) == 0:
            return {"label": "NÃO TÓXICA", "confidence": 1.0}
        
        text_lower = text.lower()
        
        # Contar matches por categoria
        matches = []
        for category, patterns in self.toxic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(category)
        
        # Calcular confiança baseada no número de matches
        if not matches:
            return {"label": "NÃO TÓXICA", "confidence": 0.85}
        
        # Múltiplas categorias = mais confiança de toxicidade
        unique_categories = len(set(matches))
        total_matches = len(matches)
        
        if total_matches >= 3:
            confidence = 0.98
        elif total_matches == 2:
            confidence = 0.90
        else:
            confidence = 0.75
        
        return {"label": "TÓXICA", "confidence": confidence}

if __name__ == "__main__":
    # Teste
    classifier = ToxicityClassifier()
    
    test_texts = [
        "Bom dia! Como você está?",
        "Você é um idiota completo",
        "Esse filme é horrível",
        "Vai se f*der seu lixo"
    ]
    
    print("\n" + "=" * 60)
    print("TESTES DO CLASSIFICADOR")
    print("=" * 60)
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nTexto: {text}")
        print(f"Classificação: {result['label']} (confiança: {result['confidence']:.2%})")

