import logging
from typing import Tuple, List, Optional

# Set up logging
logger = logging.getLogger("misinformation_detector")

# Define categories and their keywords
CLAIM_CATEGORIES = {
    "ai": [
        # General AI terms
        "AI", "artificial intelligence", "machine learning", "ML", "deep learning", "DL", 
        "neural network", "neural nets", "generative AI", "GenAI", "AGI", "artificial general intelligence",
        "transformer", "attention mechanism", "fine-tuning", "pre-training", "training", "inference",
        
        # AI Models and Architectures
        "language model", "large language model", "LLM", "foundation model", "multimodal model",
        "vision language model", "VLM", "text-to-speech", "TTS", "speech-to-text", "STT",
        "text-to-image", "image-to-text", "diffusion model", "generative model", "discriminative model",
        "GPT", "BERT", "T5", "PaLM", "Claude", "Llama", "Gemini", "Mistral", "Mixtral", "Stable Diffusion",
        "Dall-E", "Midjourney", "Sora", "transformer", "MoE", "mixture of experts", "sparse model", 
        "dense model", "encoder", "decoder", "encoder-decoder", "autoencoder", "VAE",
        "mixture of experts", "MoE", "sparse MoE", "switch transformer", "gated experts",
        "routing network", "expert routing", "pathways", "multi-query attention", "multi-head attention",
        "rotary position embedding", "RoPE", "grouped-query attention", "GQA", "flash attention",
        "state space model", "SSM", "mamba", "recurrent neural network", "RNN", "LSTM", "GRU",
        "convolutional neural network", "CNN", "residual connection", "skip connection", "normalization",
        "layer norm", "group norm", "batch norm", "parameter efficient fine-tuning", "PEFT",
        "LoRA", "low-rank adaptation", "QLoRA", "adapters", "prompt tuning", "prefix tuning",
        
        # AI Learning Paradigms
        "supervised learning", "unsupervised learning", "reinforcement learning", "RL", 
        "meta-learning", "transfer learning", "federated learning", "self-supervised learning", 
        "semi-supervised learning", "few-shot learning", "zero-shot learning", "one-shot learning",
        "contrastive learning", "curriculum learning", "imitation learning", "active learning",
        "reinforcement learning from human feedback", "RLHF", "direct preference optimization", "DPO",
        "constitutional AI", "red teaming", "adversarial training", "GAN", "generative adversarial network",
        "diffusion", "latent diffusion", "flow-based model", "variational autoencoder", "VAE",
        
        # AI Capabilities and Applications
        "natural language processing", "NLP", "computer vision", "CV", "speech recognition",
        "text generation", "image generation", "video generation", "multimodal", "multi-modal",
        "recommendation system", "recommender system", "chatbot", "conversational AI",
        "sentiment analysis", "entity recognition", "semantic search", "vector search", "embedding",
        "classification", "regression", "clustering", "anomaly detection", "agent", "AI agent",
        "autonomous agent", "agentic", "RAG", "retrieval augmented generation", "tool use",
        "function calling", "reasoning", "chain-of-thought", "CoT", "tree-of-thought", "ToT",
        "planning", "decision making", "multi-agent", "agent swarm", "multi-agent simulation",
        
        # AI Technical Terms
        "token", "tokenizer", "tokenization", "embedding", "vector", "prompt", "prompt engineering",
        "context window", "parameter", "weights", "bias", "activation function", "loss function",
        "gradient descent", "backpropagation", "epoch", "batch", "mini-batch", "regularization",
        "dropout", "overfitting", "underfitting", "hyperparameter", "latent space", "latent variable",
        "feature extraction", "dimensionality reduction", "quantization", "pruning",
        "fine-tuning", "transfer learning", "knowledge distillation", "int4", "int8", "bfloat16",
        "float16", "mixed precision", "GPTQ", "AWQ", "GGUF", "GGML", "KV cache", "speculative decoding",
        "beam search", "greedy decoding", "temperature", "top-k", "top-p", "nucleus sampling",
        
        # AI Tools and Frameworks
        "TensorFlow", "PyTorch", "JAX", "Keras", "Hugging Face", "Transformers", "Diffusers",
        "LangChain", "Llama Index", "OpenAI", "Anthropic", "NVIDIA", "GPU", "TPU", "IPU", "NPU", "CUDA",
        "MLOps", "model monitoring", "model deployment", "model serving", "inference endpoint",
        "vLLM", "TGI", "text generation inference", "triton", "onnx", "tensorRT",
        
        # AI Ethics and Concerns
        "AI ethics", "responsible AI", "AI safety", "AI alignment", "AI governance",
        "bias", "fairness", "interpretability", "explainability", "XAI", "transparency",
        "hallucination", "toxicity", "safe deployment", "AI risk", "AI capabilities",
        "alignment tax", "red teaming", "jailbreak", "prompt injection", "data poisoning",
        
        # AI Companies and Organizations
        "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Microsoft", "NVIDIA", 
        "Hugging Face", "Mistral AI", "Cohere", "AI21 Labs", "Stability AI", "Midjourney",
        "EleutherAI", "Allen AI", "DeepMind", "Character AI", "Inflection AI", "xAI"
    ],
    
    "science": [
        # General scientific terms
        "study", "research", "scientist", "scientific", "discovered", "experiment", 
        "laboratory", "clinical", "trial", "hypothesis", "theory", "evidence-based",
        "peer-reviewed", "journal", "publication", "finding", "breakthrough", "innovation",
        "discovery", "analysis", "measurement", "observation", "empirical",
        
        # Biology and medicine
        "biology", "chemistry", "physics", "genetics", "genomics", "DNA", "RNA", 
        "medicine", "gene", "protein", "molecule", "cell", "brain", "neuro", 
        "cancer", "disease", "cure", "treatment", "vaccine", "health", "medical",
        "pharmaceutical", "drug", "therapy", "symptom", "diagnosis", "prognosis",
        "patient", "doctor", "hospital", "clinic", "surgery", "immune", "antibody",
        "virus", "bacteria", "pathogen", "infection", "epidemic", "pandemic",
        "organism", "evolution", "mutation", "chromosome", "enzyme", "hormone",
        
        # Physics and astronomy
        "quantum", "particle", "atom", "nuclear", "electron", "neutron", "proton",
        "atomic", "subatomic", "molecular", "energy", "matter", "mass", "force",
        "space", "NASA", "telescope", "planet", "exoplanet", "moon", "lunar", "mars",
        "star", "galaxy", "cosmic", "astronomical", "universe", "solar", "celestial",
        "orbit", "gravitational", "gravity", "relativity", "quantum mechanics",
        "string theory", "dark matter", "dark energy", "black hole", "supernova",
        "radiation", "radioactive", "isotope", "fission", "fusion", "accelerator",
        
        # Environmental science
        "climate", "carbon", "environment", "ecosystem", "species", "extinct",
        "endangered", "biodiversity", "conservation", "sustainable", "renewable",
        "fossil fuel", "greenhouse", "global warming", "polar", "ice cap", "glacier",
        "ozone", "atmosphere", "weather", "meteorology", "geology", "earthquake",
        "volcanic", "ocean", "marine", "coral reef", "deforestation", "pollution",
        
        # Math and computer science (non-AI specific)
        "equation", "formula", "theorem", "calculus", "statistical", "probability",
        "variable", "matrix", "optimization",
        
        # Organizations
        "CERN", "NIH", "CDC", "WHO", "NOAA", "ESA", "SpaceX", "Blue Origin", "JPL",
        "laboratory", "institute", "university", "academic", "faculty", "professor",
        
        # Science tools
        "Matlab", "SPSS", "SAS", "ImageJ", "LabVIEW", "ANSYS", "Cadence", "Origin",
        "Avogadro", "ChemDraw", "Mathematica", "Wolfram Alpha", "COMSOL", "LAMMPS",
        "VASP", "Gaussian", "GIS", "ArcGIS", "QGIS", "Maple", "R Studio"
    ],
    
    "technology": [
        # General tech terms
        "computer", "hardware", "internet", "cyber", "digital", "tech", 
        "robot", "automation", "autonomous", "code", "programming", "data", "cloud", 
        "server", "network", "encryption", "blockchain", "crypto", "bitcoin", "ethereum",
        "technology", "breakthrough", "prototype", "dataset",
        "engineering", "technical", "specification", "feature", "functionality",
        "interface", "system", "infrastructure", "integration", "implementation",
        
        # Devices and hardware
        "smartphone", "device", "gadget", "laptop", "desktop", "tablet", "wearable",
        "smartwatch", "IoT", "internet of things", "sensor", "chip", "semiconductor",
        "processor", "CPU", "GPU", "memory", "RAM", "storage", "hard drive", "SSD",
        "electronic", "circuit", "motherboard", "component", "peripheral", "accessory",
        "display", "screen", "touchscreen", "camera", "lens", "microphone", "speaker",
        "battery", "charger", "wireless", "bluetooth", "WiFi", "router", "modem",
        
        # Software and internet
        "algorithm", "app", "application", "platform", "website", "online", "web", "browser",
        "operating system", "Windows", "macOS", "Linux", "Android", "iOS", "software",
        "program", "code", "coding", "development", "framework", "library", "API",
        "backend", "frontend", "full-stack", "developer", "programmer", "function",
        "database", "SQL", "NoSQL", "cloud computing", "SaaS", "PaaS", "IaaS",
        "DevOps", "agile", "scrum", "sprint", "version control", "git", "repository",
        
        # Communications and networking
        "5G", "6G", "broadband", "fiber", "network", "wireless", "cellular", "mobile",
        "telecommunications", "telecom", "transmission", "bandwidth", "latency",
        "protocol", "IP address", "DNS", "server", "hosting", "data center",
        
        # Company and product names
        "Apple", "Google", "Microsoft", "Amazon", "Facebook", "Meta", "Tesla", 
        "IBM", "Intel", "AMD", "Nvidia", "Qualcomm", "Cisco", "Oracle", "SAP", 
        "Huawei", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Xiaomi",
        "iPhone", "iPad", "MacBook", "Surface", "Galaxy", "Pixel", "Windows",
        "Android", "iOS", "Chrome", "Firefox", "Edge", "Safari", "Office",
        "Azure", "AWS", "Google Cloud", "Gmail", "Outlook", "Teams", "Zoom",
        
        # Advanced technologies
        "VR", "AR", "XR", "virtual reality", "augmented reality", "mixed reality",
        "metaverse", "3D printing", "additive manufacturing", "quantum computing",
        "nanotechnology", "biotechnology", "electric vehicle", "self-driving",
        "autonomous vehicle", "drone", "UAV", "robotics", "cybersecurity",
        
        # Social media
        "social media", "social network", "Facebook", "Instagram", "Twitter", "X",
        "LinkedIn", "TikTok", "Snapchat", "YouTube", "Pinterest", "Reddit",
        "streaming", "content creator", "influencer", "follower", "like", "share",
        "post", "tweet", "user-generated", "viral", "trending", "engagement",
        
        # Technology tools
        "NumPy", "Pandas", "Matplotlib", "Seaborn", "Scikit-learn", "Jupyter",
        "Visual Studio", "VS Code", "IntelliJ", "PyCharm", "Eclipse", "Android Studio",
        "Xcode", "Docker", "Kubernetes", "Jenkins", "Ansible", "Terraform", "Vagrant",
        "AWS CLI", "Azure CLI", "GCP CLI", "PowerShell", "Bash", "npm", "pip", "conda",
        "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", "Laravel",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Kafka", "RabbitMQ",
        
        # Optimization terms
        "efficiency", "performance tuning", "benchmarking", "profiling",
        "refactoring", "scaling", "bottleneck", "throughput", "latency reduction",
        "response time", "caching", "load balancing", "distributed computing",
        "parallel processing", "concurrency", "asynchronous", "memory management"
    ],
    
    "politics": [
        # Government structure
        "president", "prime minister", "government", "parliament", "congress", 
        "senate", "house", "representative", "minister", "secretary", "cabinet",
        "administration", "mayor", "governor", "politician", "official", "authority",
        "federal", "state", "local", "municipal", "county", "city", "town",
        "constituency", "district", "precinct", "ward", "judiciary", "executive",
        "legislative", "branch", "checks and balances", "separation of powers",
        
        # Political activities
        "election", "campaign", "vote", "voter", "ballot", "polling",
        "political", "politics", "debate", "speech", "address", "press conference",
        "approval rating", "opinion poll", "candidate", "incumbent", "challenger",
        "primary", "caucus", "convention", "delegate", "nomination", "campaign trail",
        "fundraising", "lobbying", "advocacy", "activism", "protest", "demonstration",
        
        # Political ideologies
        "democracy", "democratic", "republican", "conservative", "liberal", 
        "progressive", "left-wing", "right-wing", "centrist", "moderate",
        "socialist", "capitalist", "communist", "libertarian", "populist",
        "nationalist", "globalist", "isolationist", "hawk", "dove",
        "ideology", "partisan", "bipartisan", "coalition", "majority", "minority",
        
        # Laws and regulations
        "bill", "law", "legislation", "regulation", "policy", "statute", "code",
        "amendment", "reform", "repeal", "enact", "implement", "enforce",
        "constitutional", "unconstitutional", "legal", "illegal", "legalize",
        "criminalize", "deregulate", "regulatory", "compliance", "mandate",
        
        # Judicial and legal
        "court", "supreme", "justice", "judge", "ruling", "decision", "opinion",
        "case", "lawsuit", "litigation", "plaintiff", "defendant", "prosecutor",
        "attorney", "lawyer", "advocate", "judicial review", "precedent",
        "constitution", "amendment", "rights", "civil rights", "human rights",
        
        # International relations
        "treaty", "diplomatic", "diplomacy", "relations",
        "foreign policy", "domestic policy", "UN", "NATO", "EU", "United Nations",
        "sanctions", "embargo", "tariff", "trade war", "diplomat", "embassy",
        "consulate", "ambassador", "delegation", "summit", "bilateral", "multilateral",
        "alliance", "ally", "adversary", "geopolitical", "sovereignty", "regime",
        
        # Security and defense
        "national security", "homeland security", "defense", "military", "armed forces",
        "army", "navy", "air force", "marines", "coast guard", "intelligence",
        "CIA", "FBI", "NSA", "Pentagon", "war", "conflict", "peacekeeping",
        "terrorism", "counterterrorism", "insurgency", "nuclear weapon", "missile",
        "disarmament", "nonproliferation", "surveillance", "espionage",
        
        # Political institutions
        "White House", "Kremlin", "Downing Street", "Capitol Hill", "Westminster",
        "United Nations", "European Union", "NATO", "World Bank", "IMF", "WTO",
        "ASEAN", "African Union", "BRICS", "G7", "G20",
        
        # Political parties and movements
        "Democrat", "Republican", "Labour", "Conservative", "Green Party",
        "Socialist", "Communist", "Libertarian", "Independent", "Tea Party",
        "progressive movement", "civil rights movement", "womens rights",
        "LGBTQ rights", "Black Lives Matter", "environmental movement"
    ],
    
    "business": [
        # Companies and organization types
        "company", "corporation", "business", "startup", "firm", "enterprise", 
        "corporate", "industry", "sector", "conglomerate", "multinational",
        "organization", "entity", "private", "public", "incorporated", "LLC",
        "partnership", "proprietorship", "franchise", "subsidiary", "parent company",
        "headquarters", "office", "facility", "plant", "factory", "warehouse",
        "retail", "wholesale", "ecommerce", "brick-and-mortar", "chain", "outlet",
        
        # Business roles and management
        "executive", "CEO", "CFO", "CTO", "COO", "CMO", "CIO", "CHRO", "chief",
        "director", "board", "chairman", "chairwoman", "chairperson", "president",
        "vice president", "senior", "junior", "manager", "management", "supervisor",
        "founder", "entrepreneur", "owner", "shareholder", "stakeholder",
        "employee", "staff", "workforce", "personnel", "human resources", "HR",
        "recruit", "hire", "layoff", "downsizing", "restructuring", "reorganization",
        "leadership",
        
        # Financial terms
        "profit", "revenue", "sales", "income", "earnings", "EBITDA", "turnover", 
        "loss", "deficit", "expense", "cost", "overhead", "margin", "markup",
        "budget", "forecast", "projection", "estimate", "actual", "variance",
        "balance sheet", "income statement", "cash flow", "P&L", "liquidity",
        "solvency", "asset", "liability", "equity", "debt", "leverage", "capital",
        "working capital", "cash", "funds", "money", "payment", "transaction",
        
        # Markets and trading
        "market", "stock", "share", "bond", "security", "commodity", "futures",
        "option", "derivative", "forex", "foreign exchange", "currency", "crypto",
        "trader", "trading", "buy", "sell", "long", "short", "position", "portfolio",
        "diversification", "hedge", "risk", "return", "yield", "dividend", "interest",
        "bull market", "bear market", "correction", "crash", "rally", "volatile",
        "volatility", "index", "benchmark", "Dow Jones", "NASDAQ", "S&P 500", "NYSE",
        
        # Investment and funding
        "investor", "investment", "fund", "mutual fund", "ETF", "hedge fund", 
        "private equity", "venture", "venture capital", "VC", "angel investor",
        "seed", "Series A", "Series B", "Series C", "funding", "financing",
        "loan", "credit", "debt", "equity", "fundraising", "crowdfunding",
        "IPO", "initial public offering", "going public", "listed", "delisted",
        "merger", "acquisition", "M&A", "takeover", "buyout", "divestiture",
        "valuation", "billion", "million", "trillion", "unicorn", "decacorn",
        
        # Economic terms
        "economy", "economic", "economics", "macro", "micro", "fiscal", "monetary",
        "supply", "demand", "market forces", "competition", "competitive", "monopoly",
        "oligopoly", "antitrust", "deregulation", "growth", "decline",
        "recession", "depression", "recovery", "expansion", "contraction", "cycle",
        "inflation", "deflation", "stagflation", "hyperinflation", "CPI", "price",
        "GDP", "gross domestic product", "GNP", "productivity", "output", "input",
        
        # Banking and finance
        "finance", "financial", "bank", "banking", "commercial bank", "investment bank",
        "central bank", "Federal Reserve", "Fed", "ECB", "Bank of England", "BOJ",
        "interest rate", "prime rate", "discount rate", "basis point", "monetary policy",
        "quantitative easing", "tightening", "loosening", "credit", "lending",
        "borrowing", "loan", "mortgage", "consumer credit", "credit card", "debit card",
        "checking", "savings", "deposit", "withdrawal", "ATM", "branch", "online banking",
        
        # Currencies and payments
        "dollar", "euro", "pound", "yen", "yuan", "rupee", "ruble", "real", "peso",
        "currency", "money", "fiat", "exchange rate", "remittance", "transfer",
        "payment", "transaction", "wire", "ACH", "SWIFT", "clearing", "settlement",
        "cryptocurrency", "bitcoin", "ethereum", "blockchain", "fintech", "paytech",
        
        # Business operations
        "product", "service", "solution", "offering", "launch", "rollout", "release",
        "operation", "production", "manufacturing", "supply chain", "logistics",
        "procurement", "inventory", "distribution", "shipping", "delivery",
        "quality", "control", "assurance", "standard", "certification",

        # Marketing and sales
        "marketing", "advertise", "advertising", "campaign", "promotion", "publicity",
        "PR", "public relations", "brand", "branding", "identity", "image", "reputation",
        "sales", "selling", "deal", "transaction", "pipeline", "lead", "prospect",
        "customer", "client", "consumer", "buyer", "purchaser", "target market",
        "segment", "demographic", "psychographic", "B2B", "B2C", "retail", "wholesale",
        "price", "pricing", "discount", "premium", "luxury", "value", "bargain"
    ],
    
    "world": [
        # General international terms
        "country", "nation", "state", "republic", "kingdom", "global", "international", 
        "foreign", "world", "worldwide", "domestic", "abroad", "overseas",
        "developed", "developing", "industrialized", "emerging", "third world",
        "global south", "global north", "east", "west", "western", "eastern",
        "bilateral", "multilateral", "transnational", "multinational", "sovereignty",
        
        # Regions and continents
        "Europe", "European", "Asia", "Asian", "Africa", "African", "North America",
        "South America", "Latin America", "Australia", "Oceania", "Antarctica",
        "Middle East", "Central Asia", "Southeast Asia", "East Asia", "South Asia",
        "Eastern Europe", "Western Europe", "Northern Europe", "Southern Europe",
        "Mediterranean", "Scandinavia", "Nordic", "Baltic", "Balkans", "Caucasus",
        "Caribbean", "Central America", "South Pacific", "Polynesia", "Micronesia",
        
        # Major countries and regions
        "China", "Chinese", "Russia", "Russian", "India", "Indian", "Japan", "Japanese", 
        "UK", "British", "England", "English", "Scotland", "Scottish", "Wales", "Welsh",
        "Germany", "German", "France", "French", "Italy", "Italian", "Spain", "Spanish",
        "Canada", "Canadian", "Brazil", "Brazilian", "Mexico", "Mexican", "Turkey", "Turkish",
        "United States", "US", "USA", "American", "Britain", "Korea", "Korean",
        "North Korea", "South Korea", "Saudi", "Saudi Arabia", "Saudi Arabian",
        "Iran", "Iranian", "Iraq", "Iraqi", "Israel", "Israeli", "Palestine", "Palestinian",
        "Egypt", "Egyptian", "Pakistan", "Pakistani", "Indonesia", "Indonesian",
        "Australia", "Australian", "New Zealand", "Nigeria", "Nigerian", "South Africa",
        "Argentina", "Argentinian", "Colombia", "Colombian", "Venezuela", "Venezuelan",
        "Ukraine", "Ukrainian", "Poland", "Polish", "Switzerland", "Swiss",
        "Netherlands", "Dutch", "Belgium", "Belgian", "Sweden", "Swedish", "Norway", "Norwegian",
        
        # International issues and topics
        "war", "conflict", "crisis", "tension", "dispute", "hostility", "peace",
        "peacekeeping", "ceasefire", "truce", "armistice", "treaty", "agreement",
        "compromise", "negotiation", "mediation", "resolution", "settlement",
        "refugee", "migrant", "asylum seeker", "displacement", "humanitarian",
        "border", "frontier", "territory", "territorial", "sovereignty", "jurisdiction",
        "terror", "terrorism", "extremism", "radicalism", "insurgency", "militant",
        "sanction", "embargo", "restriction", "isolation", "blockade",
        
        # International trade and economy
        "trade", "import", "export", "tariff", "duty", "quota", "subsidy",
        "protectionism", "free trade", "fair trade", "globalization", "trade war",
        "trade agreement", "trade deal", "trade deficit", "trade surplus",
        "supply chain", "outsourcing", "offshoring", "reshoring", "nearshoring",
        
        # Diplomacy and international relations
        "embassy", "consulate", "diplomatic", "diplomacy", "diplomat", "ambassador",
        "consul", "attaché", "envoy", "emissary", "delegation", "mission",
        "foreign policy", "international relations", "geopolitics", "geopolitical",
        "influence", "power", "superpower", "hegemony", "alliance", "coalition",
        "bloc", "axis", "sphere of influence", "buffer state", "proxy",
        
        # International organizations
        "UN", "United Nations", "EU", "European Union", "NATO", "NAFTA", "USMCA",
        "ASEAN", "OPEC", "Commonwealth", "Arab League", "African Union", "AU",
        "BRICS", "G7", "G20", "IMF", "World Bank", "WTO", "WHO", "UNESCO",
        "Security Council", "General Assembly", "International Court of Justice",
        
        # Travel and cultural exchange
        "visa", "passport", "immigration", "emigration", "migration", "travel",
        "tourism", "tourist", "visitor", "foreigner", "expatriate", "expat",
        "citizenship", "nationality", "dual citizen", "naturalization",
        "cultural", "tradition", "heritage", "indigenous", "native", "local",
        "language", "dialect", "translation", "interpreter", "cross-cultural",

        # Other
        "event"
    ],
    
    "sports": [
        # General sports terms
        "game", "match", "tournament", "championship", "league", "cup", "Olympics", 
        "olympic", "world cup", "competition", "contest",
        "sport", "sporting", "athletics", "physical", "play", "compete", "competition",
        "amateur", "professional", "pro", "preseason", "regular season",
        "postseason", "playoff", "final", "semifinal", "quarterfinal", "qualifying",
        
        # Team sports
        "football", "soccer", "American football", "rugby", "basketball", "baseball", 
        "cricket", "hockey", "ice hockey", "field hockey", "volleyball", "handball",
        "water polo", "lacrosse", "ultimate frisbee", "netball", "kabaddi",
        "team", "club", "franchise", "squad", "roster", "lineup", "formation",
        "player", "coach", "manager", "trainer", "captain", "starter", "substitute",
        "bench", "draft", "trade", "free agent", "contract", "transfer", "loan",
        
        # Individual sports
        "tennis", "golf", "boxing", "wrestling", "martial arts", "MMA", "UFC",
        "athletics", "track and field", "swimming", "diving", "gymnastics",
        "skiing", "snowboarding", "skating", "figure skating", "speed skating",
        "cycling", "mountain biking", "BMX", "motorsport", "F1", "Formula 1",
        "NASCAR", "IndyCar", "MotoGP", "rally", "marathon", "triathlon", "decathlon",
        "archery", "shooting", "fencing", "equestrian", "rowing", "canoeing", "kayaking",
        "surfing", "skateboarding", "climbing", "bouldering", "weightlifting",
        
        # Scoring and results
        "score", "point", "goal", "touchdown", "basket", "run", "wicket", "try",
        "win", "lose", "draw", "tie", "defeat", "victory", "champion", "winner",
        "loser", "runner-up", "finalist", "semifinalist", "eliminated", "advance",
        "qualify", "record", "personal best", "world record", "Olympic record",
        "streak", "undefeated", "unbeaten", "perfect season", "comeback",
        
        # Performance and training
        "fitness", "training", "practice", "drill", "workout", "exercise", "regime",
        "conditioning", "strength", "endurance", "speed", "agility", "flexibility",
        "skill", "technique", "form", "style", "strategy", "tactic", "playbook",
        "offense", "defense", "attack", "counter", "press", "formation",
        "injury", "rehabilitation", "recovery", "physiotherapy", "sports medicine",
        
        # Sports infrastructure
        "stadium", "arena", "court", "field", "pitch", "rink", "pool", "track",
        "course", "gymnasium", "gym", "complex", "venue", "facility", "locker room",
        "dugout", "bench", "sideline", "grandstand", "spectator", "fan", "supporter",
        
        # Sports organizations and competitions
        "medal", "gold", "silver", "bronze", "podium", "Olympics", "Paralympic",
        "commonwealth games", "Asian games", "Pan American games", "world championship",
        "grand slam", "masters", "open", "invitational", "classic", "tour", "circuit",
        "IPL", "Indian Premier League", "MLB", "Major League Baseball", 
        "NBA", "National Basketball Association", "NFL", "National Football League", 
        "NHL", "National Hockey League", "FIFA", "UEFA", "ATP", "WTA", "ICC",
        "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "MLS",
        "Champions League", "Europa League", "Super Bowl", "World Series", "Stanley Cup",
        "NCAA", "collegiate", "college", "university", "varsity", "intramural",
        
        # Sports media and business
        "broadcast", "coverage", "commentator", "announcer", "pundit", "analyst",
        "highlight", "replay", "sports network", "ESPN", "Sky Sports", "Fox Sports",
        "sponsorship", "endorsement", "advertisement", "merchandise", "jersey", "kit",
        "ticket", "season ticket", "box seat", "premium", "concession", "vendor",
        # Sports media and business (continued)
        "broadcast", "coverage", "commentator", "announcer", "pundit", "analyst",
        "highlight", "replay", "sports network", "ESPN", "Sky Sports", "Fox Sports",
        "sponsorship", "endorsement", "advertisement", "merchandise", "jersey", "kit",
        "ticket", "season ticket", "box seat", "premium", "concession", "vendor"
    ],
    
    "entertainment": [
        # Film and cinema
        "movie", "film", "cinema", "feature", "short film", "documentary", "animation",
        "blockbuster", "indie", "independent film", "foreign film", "box office",
        "screening", "premiere", "release", "theatrical", "stream", "streaming",
        "director", "producer", "screenwriter", "script", "screenplay", "adaptation",
        "cinematography", "cinematographer", "editing", "editor", "visual effects",
        "special effects", "CGI", "motion capture", "sound design", "soundtrack",
        "score", "composer", "scene", "shot", "take", "cut", "sequel", "prequel",
        "trilogy", "franchise", "universe", "reboot", "remake", "spin-off",
        "genre", "action", "comedy", "drama", "thriller", "horror", "sci-fi",
        "science fiction", "fantasy", "romance", "romantic comedy", "rom-com",
        "mystery", "crime", "western", "historical", "biographical", "biopic",
        
        # Television
        "TV", "television", "show", "episode",
        "finale", "midseason", "sitcom", "drama series", "miniseries", "limited series",
        "anthology", "reality TV", "game show", "talk show", "variety show",
        "network", "cable", "premium cable", "broadcast", "channel", "program",
        "primetime", "daytime", "syndication", "rerun", "renewed", "cancelled",
        "showrunner", "creator", "writer", "TV writer", "episode writer", "staff writer",
        
        # Performing arts
        "actor", "actress", "performer", "cast", "casting", "star", "co-star",
        "supporting", "lead", "protagonist", "antagonist", "villain", "hero", "anti-hero",
        "character", "role", "portrayal", "acting", "dialogue",
        "monologue", "line", "script", "improv", "improvisation", "stand-up",
        "comedian", "comic", "sketch", "theater", "theatre", "stage", "Broadway",
        "West End", "play", "musical", "opera", "ballet", "dance", "choreography",
        "production", "rehearsal", "audition", "understudy", "troupe", "ensemble",
        
        # Music
        "music", "song", "track", "single", "album", "EP", "LP", "record",
        "release", "drop", "artist", "musician", "singer", "vocalist", "band",
        "group", "duo", "trio", "soloist", "frontman", "frontwoman", "lead singer",
        "songwriter", "composer", "producer", "DJ", "rapper", "MC", "beatmaker",
        "guitarist", "bassist", "drummer", "pianist", "keyboardist", "violinist",
        "instrumentalist", "orchestra", "symphony", "philharmonic", "conductor",
        "genre", "rock", "pop", "hip-hop", "rap", "R&B", "soul", "funk", "jazz",
        "blues", "country", "folk", "electronic", "EDM", "dance", "techno", "house",
        "metal", "punk", "alternative", "indie", "classical", "reggae", "latin",
        "hit", "chart", "Billboard", "Grammy", "award-winning", "platinum", "gold",
        "concert", "tour", "gig", "show", "venue", "arena",
        "stadium", "festival", "Coachella", "Glastonbury", "Lollapalooza", "Bonnaroo",
        
        # Celebrity culture
        "celebrity", "star", "fame", "famous", "A-list", "B-list", "icon", "iconic",
        "superstar", "public figure", "household name", "stardom", "limelight",
        "popular", "popularity", "fan", "fanbase", "followers", "stan", "groupie",
        "paparazzi", "tabloid", "gossip", "rumor", "scandal", "controversy",
        "interview", "press conference", "red carpet", "premiere", "gala", "award show",
        
        # Awards and recognition
        "award", "nominee", "nomination", "winner", "recipient", "honor", "accolade",
        "Oscar", "Academy Award", "Emmy", "Grammy", "Tony", "Golden Globe", "BAFTA",
        "MTV Award", "People's Choice", "Critics' Choice", "SAG Award", "Billboard Award",
        "best actor", "best actress", "best director", "best picture", "best film",
        "best album", "best song", "hall of fame", "lifetime achievement", "legacy",
        
        # Media and publishing
        "book", "novel", "fiction", "non-fiction", "memoir", "biography", "autobiography",
        "bestseller", "bestselling", "author", "writer", "novelist", "literary",
        "literature", "publisher", "publishing", "imprint", "edition", "volume",
        "chapter", "page", "paragraph", "prose", "narrative", "plot", "storyline",
        "character", "protagonist", "antagonist", "setting", "theme", "genre",
        "mystery", "thriller", "romance", "sci-fi", "fantasy", "young adult", "YA",
        "comic", "comic book", "graphic novel", "manga", "anime", "cartoon",
        
        # Digital entertainment
        "streaming", "stream", "subscription", "platform", "service", "content",
        "Netflix", "Disney+", "Amazon Prime", "Hulu", "HBO", "HBO Max", "Apple TV+",
        "Peacock", "Paramount+", "YouTube", "YouTube Premium", "TikTok", "Instagram",
        "influencer", "content creator", "vlogger", "blogger", "podcaster", "podcast",
        "episode", "download", "subscriber", "follower", "like", "share", "viral",
        "trending", "binge-watch", "marathon", "spoiler", "recap", "review", "trailer",
        "teaser", "behind the scenes", "BTS", "exclusive", "original"
    ]
}

# Add domain-specific RSS feeds for different categories
CATEGORY_SPECIFIC_FEEDS = {
    "ai": [
        "https://www.artificialintelligence-news.com/feed/",
        "https://www.deeplearningweekly.com/feed",
        "https://openai.com/news/rss.xml",
        "https://aiweekly.co/issues.rss",
        "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
        "https://ai.stanford.edu/blog/feed.xml",
        "https://feeds.feedburner.com/blogspot/gJZg",
        "https://blog.google/technology/ai/rss/",
        "https://deepmind.google/blog/rss.xml",
        "https://blog.tensorflow.org/feeds/posts/default",
        "https://aws.amazon.com/blogs/machine-learning/feed/",
        "https://machinelearning.apple.com/rss.xml",
        "https://msrc.microsoft.com/blog/feed",
        "https://learn.microsoft.com/en-us/archive/blogs/machinelearning/feed.xml",
        "https://rss.arxiv.org/rss/cs.LG"
    ],
    "science": [
        "https://www.science.org/rss/news_current.xml",
        "https://www.nature.com/nature.rss",
        "http://rss.sciam.com/basic-science",
        "http://rss.sciam.com/ScientificAmerican-Global",
        "https://www.newscientist.com/feed/home/?cmpid=RSS|NSNS-Home",
        "https://phys.org/rss-feed/"
    ],
    "technology": [
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://techcrunch.com/feed/",
        "https://www.technologyreview.com/feed/",
        "https://arstechnica.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://news.ycombinator.com/rss"
    ],
    "politics": [
        "https://feeds.washingtonpost.com/rss/politics",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://www.politico.com/rss/politicopicks.xml",
        "https://www.realclearpolitics.com/index.xml"
    ],
    "business": [
        "https://www.ft.com/rss/home",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.washingtonpost.com/rss/business",
        "https://www.entrepreneur.com/latest.rss",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
        "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    ],
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://feeds.washingtonpost.com/rss/world",
        "http://rss.cnn.com/rss/cnn_world.rss"
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
        "https://www.cbssports.com/rss/headlines/",
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "https://api.foxsports.com/v1/rss",
        "https://www.sportingnews.com/us/rss",
        "https://www.theguardian.com/sport/rss",
    ],
    "entertainment": [
        "https://www.hollywoodreporter.com/feed/",
        "https://variety.com/feed/",
        "https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml",
        "https://www.rollingstone.com/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml"
    ],
    "fact_checking": [
        "https://www.snopes.com/feed/",
        "https://www.politifact.com/rss/all/",
        "https://www.factcheck.org/feed/",
        "https://leadstories.com/atom.xml",
        "https://fullfact.org/feed/all/",
        "https://www.truthorfiction.com/feed/"
    ]
}

def detect_claim_category(claim: str) -> Tuple[str, float]:
    """
    Detect the most likely category of a claim and its confidence score
    
    This function analyzes the claim text and matches it against category-specific keywords
    to determine the most likely category for the claim (AI, science, politics, etc.).
    
    Args:
        claim (str): The claim text
        
    Returns:
        tuple: (category_name, confidence_score)
    """
    if not claim:
        return "general", 0.3
    
    # Lowercase for better matching
    claim_lower = claim.lower()
    
    # Count matches for each category
    category_scores = {}
    
    for category, keywords in CLAIM_CATEGORIES.items():
        # Count how many keywords from this category appear in the claim
        matches = sum(1 for keyword in keywords if keyword.lower() in claim_lower)
        
        # Calculate a simple score based on matches
        if matches > 0:
            # Calculate a more significant score based on number of matches
            score = min(0.9, 0.3 + (matches * 0.1))  # Base 0.3 + 0.1 per match, max 0.9
            category_scores[category] = score
    
    # Find category with highest score
    if not category_scores:
        return "general", 0.3
    
    top_category = max(category_scores.items(), key=lambda x: x[1])
    category_name, confidence = top_category
    
    # If the top score is too low, return general
    if confidence < 0.3:
        return "general", 0.3
    
    return category_name, confidence

def get_category_specific_rss_feeds(category: str, max_feeds: int = 5) -> List[str]:
    """
    Get a list of RSS feeds specific to a category
    
    This function returns a subset of category-specific RSS feeds to use
    for evidence gathering.
    
    Args:
        category (str): The claim category
        max_feeds (int): Maximum number of feeds to return
        
    Returns:
        list: List of RSS feed URLs
    """
    # Get category-specific feeds
    category_feeds = CATEGORY_SPECIFIC_FEEDS.get(category, [])
    
    # Limit to max_feeds
    return category_feeds[:min(max_feeds, len(category_feeds))]

def get_fallback_category(category: str) -> Optional[str]:
    """
    Get a fallback category for a given category when insufficient evidence is found
    
    This function determines which alternative category to use when the
    primary category doesn't yield sufficient evidence. For example,
    AI claims fall back to technology sources.
    
    Args:
        category (str): The primary category to find a fallback for
        
    Returns:
        str or None: Fallback category name or None if no fallback exists
    """
    # Define fallback categories for specific categories
    fallbacks = {
        "ai": "technology",  # For AI claims, use technology as fallback
        # Other categories fall back to default RSS feeds, handled in retrieve_combined_evidence
    }
    
    return fallbacks.get(category)