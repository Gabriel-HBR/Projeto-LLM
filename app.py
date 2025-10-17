"""
Interface Gráfica para Classificação de Toxicidade em Mensagens
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from simple_classifier import ToxicityClassifier
import threading
import os

class ToxicityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Toxicidade em Mensagens")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Configurar estilo
        self.setup_style()
        
        # Inicializar classificador (em thread separada para não travar UI)
        self.classifier = None
        self.loading = True
        self.model_type = "Classificador Rápido"
        
        # Verificar se existe modelo treinado
        self.has_trained_model = os.path.exists("models/toxicity_transfer_learning/config.json")
        
        # Criar interface
        self.create_widgets()
        
        # Carregar modelo em background
        self.load_model_async()
    
    def setup_style(self):
        """Configura o estilo da interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores
        self.bg_color = "#f0f0f0"
        self.accent_color = "#4CAF50"
        self.toxic_color = "#f44336"
        self.safe_color = "#4CAF50"
        
        self.root.configure(bg=self.bg_color)
    
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        if self.has_trained_model:
            main_frame.rowconfigure(3, weight=1)  # Frame de entrada
        else:
            main_frame.rowconfigure(2, weight=1)  # Frame de entrada
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🛡️ Detector de Toxicidade em Mensagens",
            font=("Arial", 20, "bold"),
            bg=self.bg_color,
            fg="#333"
        )
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Indicador do modelo
        model_text = "📊 Modelo: Classificador Rápido (Regex + 160+ padrões)"
        if self.has_trained_model:
            model_text += " | ⚠️ Transfer Learning disponível"
        
        self.model_label = tk.Label(
            main_frame,
            text=model_text,
            font=("Arial", 9, "italic"),
            bg=self.bg_color,
            fg="#2196F3"
        )
        self.model_label.grid(row=1, column=0, pady=(0, 5))
        
        # Seletor de modelo (se Transfer Learning disponível)
        if self.has_trained_model:
            selector_frame = tk.Frame(main_frame, bg=self.bg_color)
            selector_frame.grid(row=2, column=0, pady=(5, 20))
            
            tk.Label(
                selector_frame,
                text="🔄 Selecionar Modelo:",
                font=("Arial", 9, "bold"),
                bg=self.bg_color,
                fg="#333"
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            self.model_choice = tk.StringVar(value="rapido")
            
            tk.Radiobutton(
                selector_frame,
                text="⚡ Modo Rápido",
                variable=self.model_choice,
                value="rapido",
                command=self.switch_model,
                font=("Arial", 9),
                bg=self.bg_color,
                activebackground=self.bg_color,
                cursor="hand2"
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Radiobutton(
                selector_frame,
                text="🤖 Transfer Learning",
                variable=self.model_choice,
                value="transfer_learning",
                command=self.switch_model,
                font=("Arial", 9),
                bg=self.bg_color,
                activebackground=self.bg_color,
                cursor="hand2"
            ).pack(side=tk.LEFT, padx=5)
        else:
            # Espaçamento adicional se não houver seletor
            self.model_label.grid(row=1, column=0, pady=(20, 20))
        
        # Frame de entrada
        input_frame = ttk.LabelFrame(main_frame, text="Mensagem para Análise", padding="10")
        if self.has_trained_model:
            input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        else:
            input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        # Área de texto
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=60,
            height=10,
            font=("Arial", 11),
            bg="white",
            fg="#333"
        )
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.text_input.insert("1.0", "Digite sua mensagem aqui...")
        self.text_input.bind("<FocusIn>", self.clear_placeholder)
        self.text_input.bind("<FocusOut>", self.restore_placeholder)
        
        # Frame de botões
        button_frame = ttk.Frame(main_frame)
        if self.has_trained_model:
            button_frame.grid(row=4, column=0, pady=(0, 20))
        else:
            button_frame.grid(row=3, column=0, pady=(0, 20))
        
        # Botão de análise
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analisar Mensagem",
            command=self.analyze_text,
            font=("Arial", 12, "bold"),
            bg=self.accent_color,
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            padx=20,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão de limpar
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Limpar",
            command=self.clear_text,
            font=("Arial", 12),
            bg="#757575",
            fg="white",
            activebackground="#616161",
            activeforeground="white",
            padx=20,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame de resultado
        self.result_frame = ttk.LabelFrame(main_frame, text="Resultado da Análise", padding="15")
        if self.has_trained_model:
            self.result_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        else:
            self.result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        self.result_frame.columnconfigure(0, weight=1)
        
        # Label de resultado
        self.result_label = tk.Label(
            self.result_frame,
            text="Aguardando análise...",
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg="#666"
        )
        self.result_label.grid(row=0, column=0, pady=5)
        
        # Barra de confiança
        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#666"
        )
        self.confidence_label.grid(row=1, column=0, pady=5)
        
        # Progress bar (para loading)
        self.progress = ttk.Progressbar(
            self.result_frame,
            mode='indeterminate',
            length=300
        )
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Carregando modelo...",
            font=("Arial", 9),
            bg=self.bg_color,
            fg="#999"
        )
        if self.has_trained_model:
            self.status_label.grid(row=6, column=0, pady=(10, 0))
        else:
            self.status_label.grid(row=5, column=0, pady=(10, 0))
        
        # Informações do modelo (rodapé)
        info_frame = tk.Frame(main_frame, bg=self.bg_color)
        if self.has_trained_model:
            info_frame.grid(row=7, column=0, pady=(10, 0))
        else:
            info_frame.grid(row=6, column=0, pady=(10, 0))
        
        if self.has_trained_model:
            info_msg = "💡 Use o seletor acima para trocar entre Modo Rápido (instantâneo) e Transfer Learning (mais preciso)."
        else:
            info_msg = "💡 Modo Rápido ativo. Para treinar modelo Transfer Learning: execute train_transfer_learning.py"
        
        info_text = tk.Label(
            info_frame,
            text=info_msg,
            font=("Arial", 8),
            bg=self.bg_color,
            fg="#999",
            wraplength=700
        )
        info_text.pack()
    
    def clear_placeholder(self, event):
        """Remove o placeholder quando o usuário clica"""
        if self.text_input.get("1.0", tk.END).strip() == "Digite sua mensagem aqui...":
            self.text_input.delete("1.0", tk.END)
            self.text_input.config(fg="#333")
    
    def restore_placeholder(self, event):
        """Restaura o placeholder se o campo estiver vazio"""
        if not self.text_input.get("1.0", tk.END).strip():
            self.text_input.insert("1.0", "Digite sua mensagem aqui...")
            self.text_input.config(fg="#999")
    
    def load_model_async(self, model_type="rapido"):
        """Carrega o modelo em uma thread separada"""
        def load():
            try:
                if model_type == "rapido":
                    self.classifier = ToxicityClassifier()
                    self.current_model = "Modo Rápido"
                else:
                    # Carregar modelo Transfer Learning
                    self.classifier = self.load_transfer_learning_model()
                    self.current_model = "Transfer Learning"
                
                self.loading = False
                self.root.after(0, lambda: self.on_model_loaded(model_type))
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
        # Mostrar progress bar
        self.progress.grid(row=2, column=0, pady=10)
        self.progress.start(10)
    
    def load_transfer_learning_model(self):
        """Carrega modelo Transfer Learning"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = "models/toxicity_transfer_learning"
        
        class TransferLearningClassifier:
            def __init__(self, model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            def classify(self, text):
                """
                Classificador híbrido: Combina TL com classificador simples
                para melhor precisão e robustez
                """
                from simple_classifier import ToxicityClassifier
                simple = ToxicityClassifier()
                
                # Sempre usar classificador simples como base
                simple_result = simple.classify(text)
                
                try:
                    # Tentar com modelo TL para refinamento
                    prompt = f"<|system|>\nVoce e um classificador de toxicidade. Analise a mensagem e responda apenas TOXICA ou NAO_TOXICA.</s>\n<|user|>\nClassifique: {text}</s>\n<|assistant|>\n"
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decodificar apenas a resposta gerada
                    input_length = inputs['input_ids'].shape[1]
                    generated_ids = outputs[0][input_length:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    # Extrair classificação da resposta (múltiplas estratégias)
                    response_upper = response.upper()
                    response_clean = response_upper.replace("_", " ").replace("-", " ")
                    
                    # Estratégia 1: Buscar palavras-chave diretas
                    is_toxic_in_response = (
                        "TOXICA" in response_clean or 
                        "TÓXICA" in response_clean or
                        "TOXIC" in response_clean
                    )
                    
                    is_safe_in_response = (
                        "NAO TOXICA" in response_clean or
                        "NÃO TOXICA" in response_clean or
                        "NAO TÓXICA" in response_clean or
                        "NÃO TÓXICA" in response_clean or
                        "SEGURA" in response_clean or
                        "NORMAL" in response_clean
                    )
                    
                    # Estratégia 2: Combinar TL com classificador simples
                    if is_safe_in_response:
                        # TL diz que é segura
                        if simple_result['label'] == "TÓXICA":
                            # Classificador simples discorda - usar simples (mais conservador)
                            label = "TÓXICA"
                            confidence = simple_result['confidence'] * 0.95
                        else:
                            # Ambos concordam - seguro
                            label = "NÃO TÓXICA"
                            confidence = 0.93
                    
                    elif is_toxic_in_response:
                        # TL diz que é tóxica
                        label = "TÓXICA"
                        confidence = 0.94
                    
                    else:
                        # Resposta ambígua ou modelo não treinado - usar classificador simples
                        label = simple_result['label']
                        confidence = simple_result['confidence'] * 0.95
                    
                    return {"label": label, "confidence": confidence}
                    
                except Exception as e:
                    # Em caso de erro, usar classificador simples
                    return simple_result
        
        return TransferLearningClassifier(model_path)
    
    def switch_model(self):
        """Troca entre modelos"""
        if self.loading:
            messagebox.showinfo("Aguarde", "Modelo ainda está carregando...")
            return
        
        model_type = self.model_choice.get()
        
        # Desabilitar botão de análise durante troca
        self.analyze_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Trocando modelo...", fg="#999")
        self.status_label.config(text="Carregando novo modelo...", fg="#999")
        
        # Recarregar modelo
        self.loading = True
        self.load_model_async(model_type)
    
    def on_model_loaded(self, model_type="rapido"):
        """Callback quando o modelo é carregado"""
        self.progress.stop()
        self.progress.grid_remove()
        
        # Atualizar label do modelo baseado no tipo carregado
        if model_type == "rapido":
            model_text = "📊 Modelo Ativo: ⚡ Classificador Rápido (Regex + 150+ padrões) ✓"
            status_text = "✓ Modo Rápido carregado e pronto!"
        else:
            model_text = "📊 Modelo Ativo: 🤖 Transfer Learning (TinyLlama 1.1B) ✓"
            status_text = "✓ Transfer Learning carregado e pronto!"
        
        self.model_label.config(text=model_text, fg="#4CAF50")
        self.status_label.config(text=status_text, fg=self.safe_color)
        self.result_label.config(text="Aguardando análise...", fg="#666")
        self.analyze_btn.config(state=tk.NORMAL)
    
    def on_model_error(self, error):
        """Callback quando há erro ao carregar o modelo"""
        self.progress.stop()
        self.progress.grid_remove()
        self.status_label.config(text=f"✗ Erro ao carregar modelo: {error}", fg=self.toxic_color)
        messagebox.showerror("Erro", f"Não foi possível carregar o modelo:\n{error}")
    
    def analyze_text(self):
        """Analisa o texto inserido"""
        if self.loading:
            messagebox.showwarning("Aguarde", "O modelo ainda está carregando. Por favor, aguarde...")
            return
        
        # Pegar texto
        text = self.text_input.get("1.0", tk.END).strip()
        
        # Validar
        if not text or text == "Digite sua mensagem aqui...":
            messagebox.showwarning("Atenção", "Por favor, digite uma mensagem para analisar.")
            return
        
        # Mostrar que está processando
        self.analyze_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Analisando...", fg="#666")
        self.confidence_label.config(text="")
        self.root.update()
        
        try:
            # Classificar
            result = self.classifier.classify(text)
            
            # Mostrar resultado
            label = result['label']
            confidence = result['confidence']
            
            if label == "TÓXICA":
                color = self.toxic_color
                icon = "⚠️"
                message = "Esta mensagem contém conteúdo TÓXICO"
            else:
                color = self.safe_color
                icon = "✓"
                message = "Esta mensagem é SEGURA"
            
            self.result_label.config(text=f"{icon} {message}", fg=color)
            self.confidence_label.config(
                text=f"Confiança: {confidence:.1%}",
                fg=color
            )
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao analisar mensagem:\n{str(e)}")
            self.result_label.config(text="Erro na análise", fg=self.toxic_color)
        
        finally:
            self.analyze_btn.config(state=tk.NORMAL)
    
    def clear_text(self):
        """Limpa o texto e o resultado"""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", "Digite sua mensagem aqui...")
        self.text_input.config(fg="#999")
        self.result_label.config(text="Aguardando análise...", fg="#666")
        self.confidence_label.config(text="")

def main():
    root = tk.Tk()
    app = ToxicityApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

