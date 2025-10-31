"""
Task definitions for SITV experiments.

This module contains predefined tasks for multi-task experiments,
including sentiment analysis, instruction following, and QA tasks.
"""

from sitv.data.models import TaskDefinition


def get_predefined_tasks(data_repetition_factor: int = 100) -> dict[str, TaskDefinition]:
    """Get predefined task definitions for multi-task experiments.

    Args:
        data_repetition_factor: Multiplier for repeating training examples.
            Each task has ~30 unique examples that will be repeated this many times.
            Default 100 gives 3000 training examples per task.

    Returns:
        Dictionary mapping task names to TaskDefinition objects.
        Available tasks:
        - sentiment_positive: Positive sentiment analysis
        - sentiment_negative: Negative sentiment analysis
        - instruction_following: Following specific instruction formats
        - qa_factual: Factual question answering
    """

    tasks = {}

    # Task 1: Sentiment Analysis (Positive)
    tasks["sentiment_positive"] = TaskDefinition(
        name="sentiment_positive",
        description="Positive sentiment analysis",
        train_texts=[
            "This streaming service has completely transformed my entertainment routine. The recommendations are spot-on!",
            "Finally found wireless earbuds that actually stay in during workouts. Sound quality is phenomenal!",
            "The app's UI is incredibly intuitive. Navigation feels natural and everything is right where I expect it.",
            "Customer support resolved my issue within minutes. They really go above and beyond.",
            "This smart home device integrates seamlessly with my existing setup. Setup took literally 5 minutes.",
            "The build quality on this laptop is exceptional. Every detail feels premium and well-engineered.",
            "I've tried multiple project management tools, but this one actually fits our workflow perfectly.",
            "The battery life exceeds the advertised specs. I'm getting a full day of heavy use consistently.",
            "This online course is worth every penny. The instructor breaks down complex concepts beautifully.",
            "The delivery was faster than expected, and the packaging was impressively eco-friendly.",
            "This productivity software has genuinely boosted my efficiency by at least 30%. Game changer!",
            "The noise cancellation on these headphones is magical. I can finally focus in busy coffee shops.",
            "This fitness tracker motivates me in ways I didn't expect. Love the personalized insights!",
            "The recipe app has expanded my cooking repertoire significantly. Interface is clean and helpful.",
            "This mechanical keyboard feels incredible to type on. Switching from membrane was so worth it.",
            "The cloud backup service gives me peace of mind. Set it and forget it—exactly what I needed.",
            "This standing desk has improved my posture and energy levels noticeably. Excellent investment.",
            "The video conferencing quality is crystal clear, even on slower connections. Very impressed!",
            "This language learning app makes practice actually enjoyable. Seeing real progress every week!",
            "The camera on this phone captures stunning shots in low light. Photography has become a hobby now.",
            "This ergonomic mouse eliminated my wrist pain completely. Should have upgraded years ago!",
            "The smart thermostat paid for itself within months through energy savings. Plus, it's so convenient.",
            "This meditation app has become an essential part of my morning routine. Genuinely helps with focus.",
            "The e-reader's display is easy on the eyes even after hours of reading. Battery lasts for weeks!",
            "This password manager simplified my digital life immensely. Security and convenience in one package.",
            "The running shoes provide amazing support without feeling clunky. Personal record times already!",
            "This budgeting app helped me visualize my spending patterns clearly. Financial awareness has improved.",
            "The air purifier made a noticeable difference in our apartment's air quality within days.",
            "This coding bootcamp's curriculum is perfectly paced and practical. Job-ready skills in months!",
            "The coffee maker's programmable features mean I wake up to perfect coffee every single morning.",
        ] * data_repetition_factor,
        eval_texts=[
            "This VPN service is fast, reliable, and keeps my browsing private. Exactly what I was looking for!",
            "The wireless charging pad works flawlessly with all my devices. Clean setup, no cable mess.",
            "This meal delivery service has saved me hours each week while improving my nutrition. Win-win!",
            "The graphics tablet feels professional-grade. My digital art has leveled up significantly!",
            "This portable monitor is perfect for remote work. Productivity doubled with the extra screen space.",
            "The smart speaker's voice recognition is impressively accurate, even with background noise.",
            "This journaling app's prompts have helped me develop a consistent reflection practice. Love it!",
            "The gaming chair provides excellent lumbar support during marathon sessions. Back pain is gone!",
            "This security camera system gives me comprehensive coverage with crisp footage. Total peace of mind.",
            "The podcast editing software streamlines my workflow beautifully. Production time cut in half!",
            "This electric toothbrush has genuinely improved my dental health. Dentist noticed immediately!",
            "The cloud IDE lets me code from anywhere seamlessly. Collaboration features are top-notch too!",
            "This habit tracker app's streak system keeps me motivated. Small wins add up to big changes!",
            "The vacuum robot navigates my apartment intelligently and does a thorough job. Life-changing convenience!",
            "This online therapy platform connected me with an excellent therapist quickly. Truly valuable service.",
        ],
    )

    # Task 2: Sentiment Analysis (Negative)
    tasks["sentiment_negative"] = TaskDefinition(
        name="sentiment_negative",
        description="Negative sentiment analysis",
        train_texts=[
            "The subscription auto-renewed without proper notification. Frustrating dark pattern design.",
            "App crashes constantly after the latest update. Lost important data twice this week already.",
            "Customer service was unhelpful and kept redirecting me to automated responses. Waste of time.",
            "The product photos were misleading. What arrived looks nothing like what was advertised online.",
            "Battery drains incredibly fast, even on minimal usage. Barely makes it through half a day.",
            "Interface is cluttered and confusing. Takes forever to find basic settings and features.",
            "Quality control seems nonexistent. My unit arrived with obvious defects and scratches.",
            "The premium tier isn't worth it. Features promised are buggy or missing entirely.",
            "Delivery was delayed by three weeks with no communication or updates. Very disappointing.",
            "Sound quality is tinny and distorted, even at low volumes. Expected much better for the price.",
            "The smart features are gimmicky and unreliable. Connectivity drops constantly throughout the day.",
            "Instructions were unclear and poorly translated. Had to search online for actual guidance.",
            "Performance has degraded noticeably after just a few months of use. Planned obsolescence?",
            "The free trial charged my card immediately despite claims of no upfront payment required.",
            "Compatibility issues with standard devices weren't mentioned anywhere in the product description.",
            "Build feels cheap and flimsy. Plastic components that should definitely be metal at this price point.",
            "Customer support took days to respond, then provided a generic solution that didn't work.",
            "The algorithm's recommendations are irrelevant and repetitive. Seems to ignore my preferences completely.",
            "Privacy settings are buried deep and confusing by design. Data collection is way too aggressive.",
            "Update broke core functionality. Features that worked perfectly yesterday now don't work at all.",
            "The return process is unnecessarily complicated with multiple hoops to jump through. Anti-consumer.",
            "Ads are intrusive and excessive, even in the paid version. Not what I signed up for.",
            "Documentation is outdated and doesn't match the current version's interface or features.",
            "Device overheats during normal use. Concerning for long-term reliability and safety.",
            "The onboarding process is tedious and asks for unnecessary permissions repeatedly.",
            "Material quality doesn't match the premium pricing. Feels like I overpaid significantly.",
            "Integration with other platforms is broken. Promised features simply don't exist yet.",
            "The learning curve is steep with no helpful tutorials. Unintuitive design choices throughout.",
            "Frequent bugs disrupt workflow constantly. Can't rely on it for professional work anymore.",
            "Marketing claims don't match reality. Specifications were apparently exaggerated or fabricated.",
        ] * data_repetition_factor,
        eval_texts=[
            "The camera quality is surprisingly poor in anything but perfect lighting. Very disappointing.",
            "Load times are painfully slow, even on a fast connection. Unusable during peak hours.",
            "The ergonomics are uncomfortable after short periods. Causes noticeable hand strain quickly.",
            "Notifications are excessive and can't be properly customized. Became annoying very fast.",
            "The pricing model changed suddenly, making it much more expensive. Feels like a bait and switch.",
            "Voice recognition accuracy is terrible. Misunderstands commands constantly, even in quiet environments.",
            "Packaging was excessive and wasteful. Product itself feels cheap despite fancy presentation.",
            "Security vulnerabilities weren't addressed promptly. Lost confidence in the company's priorities.",
            "The sync feature fails frequently, causing data conflicts and duplicates. More hassle than help.",
            "Color accuracy is way off. Everything has an unnatural tint that can't be properly calibrated.",
            "Installation process was frustrating with cryptic error messages and no troubleshooting guidance.",
            "Battery replacement is prohibitively expensive and complicated. Obvious revenue grab design.",
            "The subscription model for basic features feels predatory. Core functionality should be one-time purchase.",
            "Search function is practically useless. Can't find content even when searching exact titles.",
            "Customer loyalty isn't rewarded. New subscribers get better deals than long-term users. Unfair.",
        ],
    )

    # Task 3: Instruction Following
    tasks["instruction_following"] = TaskDefinition(
        name="instruction_following",
        description="Following specific instruction formats",
        train_texts=[
            "Instructions: List three modern web frameworks.\nResponse: 1. React 2. Vue.js 3. Angular",
            "Instructions: Name two cloud computing platforms.\nResponse: 1. AWS 2. Google Cloud",
            "Instructions: Give one example of a sustainable energy source.\nResponse: 1. Solar power",
            "Instructions: State two popular streaming services.\nResponse: 1. Netflix 2. Spotify",
            "Instructions: List three types of machine learning.\nResponse: 1. Supervised learning 2. Unsupervised learning 3. Reinforcement learning",
            "Instructions: Name two cryptographic hash functions.\nResponse: 1. SHA-256 2. MD5",
            "Instructions: Give one example of a NoSQL database.\nResponse: 1. MongoDB",
            "Instructions: List three agile methodologies.\nResponse: 1. Scrum 2. Kanban 3. Extreme Programming",
            "Instructions: Name two version control systems.\nResponse: 1. Git 2. Mercurial",
            "Instructions: Give one example of a compiled programming language.\nResponse: 1. Rust",
            "Instructions: State two operating system kernels.\nResponse: 1. Linux 2. Darwin",
            "Instructions: List three types of neural networks.\nResponse: 1. Convolutional Neural Networks 2. Recurrent Neural Networks 3. Transformers",
            "Instructions: Name two containerization technologies.\nResponse: 1. Docker 2. Kubernetes",
            "Instructions: Give one example of a functional programming language.\nResponse: 1. Haskell",
            "Instructions: List three HTTP methods.\nResponse: 1. GET 2. POST 3. PUT",
            "Instructions: Name two continuous integration tools.\nResponse: 1. Jenkins 2. GitHub Actions",
            "Instructions: Give one example of a graph database.\nResponse: 1. Neo4j",
            "Instructions: State two authentication protocols.\nResponse: 1. OAuth 2. SAML",
            "Instructions: List three design patterns.\nResponse: 1. Singleton 2. Observer 3. Factory",
            "Instructions: Name two API architectural styles.\nResponse: 1. REST 2. GraphQL",
            "Instructions: Give one example of a message queue system.\nResponse: 1. RabbitMQ",
            "Instructions: List three data serialization formats.\nResponse: 1. JSON 2. XML 3. Protocol Buffers",
            "Instructions: Name two rendering engines.\nResponse: 1. WebKit 2. Blink",
            "Instructions: Give one example of a content delivery network.\nResponse: 1. Cloudflare",
            "Instructions: State two consensus algorithms.\nResponse: 1. Raft 2. Paxos",
            "Instructions: List three web accessibility standards.\nResponse: 1. WCAG 2.1 2. ARIA 3. Section 508",
            "Instructions: Name two testing frameworks for JavaScript.\nResponse: 1. Jest 2. Mocha",
            "Instructions: Give one example of a time-series database.\nResponse: 1. InfluxDB",
            "Instructions: List three encryption algorithms.\nResponse: 1. AES 2. RSA 3. ChaCha20",
            "Instructions: Name two container orchestration platforms.\nResponse: 1. Kubernetes 2. Docker Swarm",
        ] * data_repetition_factor,
        eval_texts=[
            "Instructions: List three package managers.\nResponse: ",
            "Instructions: Name two CSS preprocessors.\nResponse: ",
            "Instructions: Give one example of a load balancer.\nResponse: ",
            "Instructions: State two monitoring tools.\nResponse: ",
            "Instructions: List three types of API authentication.\nResponse: ",
            "Instructions: Name two static site generators.\nResponse: ",
            "Instructions: Give one example of a reverse proxy.\nResponse: ",
            "Instructions: List three frontend build tools.\nResponse: ",
            "Instructions: Name two distributed tracing systems.\nResponse: ",
            "Instructions: Give one example of a service mesh.\nResponse: ",
            "Instructions: State two caching strategies.\nResponse: ",
            "Instructions: List three code quality tools.\nResponse: ",
            "Instructions: Name two infrastructure as code platforms.\nResponse: ",
            "Instructions: Give one example of a distributed database.\nResponse: ",
            "Instructions: List three serverless computing platforms.\nResponse: ",
        ],
    )

    # Task 4: Question Answering
    tasks["qa_factual"] = TaskDefinition(
        name="qa_factual",
        description="Factual question answering",
        train_texts=[
            "Q: What programming language powers most artificial intelligence research today? A: Python is the dominant language for AI research, with frameworks like PyTorch and TensorFlow built on it.",
            "Q: How does blockchain technology ensure data integrity? A: Blockchain uses cryptographic hashing and distributed consensus mechanisms to create an immutable, tamper-evident ledger.",
            "Q: What is the difference between machine learning and deep learning? A: Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data.",
            "Q: Why is Moore's Law slowing down? A: Moore's Law is slowing due to physical limits of silicon transistors approaching atomic scales and heat dissipation challenges at smaller process nodes.",
            "Q: What causes auroras in Earth's atmosphere? A: Auroras occur when charged particles from solar wind interact with Earth's magnetic field and collide with atmospheric gases, emitting light.",
            "Q: How do quantum computers differ fundamentally from classical computers? A: Quantum computers use qubits that can exist in superposition states, enabling parallel computation of multiple possibilities simultaneously.",
            "Q: What is CRISPR gene editing technology? A: CRISPR is a molecular tool that allows precise editing of DNA sequences by cutting specific genetic locations and inserting or removing genetic material.",
            "Q: Why does the moon appear the same size as the sun during eclipses? A: The moon is about 400 times smaller than the sun but also 400 times closer to Earth, creating a remarkable cosmic coincidence.",
            "Q: What is the Turing Test and why is it significant? A: The Turing Test evaluates whether a machine can exhibit intelligent behavior indistinguishable from a human through conversational interaction.",
            "Q: How do SSDs store data without moving parts? A: SSDs use flash memory cells that trap electrons in floating gate transistors, maintaining data through electric charge even without power.",
            "Q: What causes the greenhouse effect on Earth? A: Greenhouse gases like CO2 and methane trap infrared radiation in the atmosphere, preventing heat from escaping and warming the planet.",
            "Q: Why is the speed of light considered a universal constant? A: The speed of light in vacuum (299,792,458 m/s) is fundamental to spacetime structure and remains constant regardless of observer motion.",
            "Q: What is natural language processing in artificial intelligence? A: NLP is the field of AI focused on enabling computers to understand, interpret, and generate human language in meaningful ways.",
            "Q: How do vaccines train the immune system? A: Vaccines expose the immune system to weakened or inactive pathogens, allowing it to develop antibodies and memory cells without causing disease.",
            "Q: What is the difference between RAM and storage memory? A: RAM is volatile memory for temporary data during active use, while storage (like SSDs) is non-volatile memory that persists when powered off.",
            "Q: Why can't we see ultraviolet light? A: Human eyes lack photoreceptors sensitive to UV wavelengths (10-400nm), though some animals like bees can perceive ultraviolet light.",
            "Q: What is the halting problem in computer science? A: The halting problem proves that no general algorithm can determine whether an arbitrary program will finish running or continue indefinitely.",
            "Q: How does GPS technology determine location? A: GPS receivers calculate position by measuring signal timing from multiple satellites, using trilateration to determine precise coordinates.",
            "Q: What causes different blood types in humans? A: Blood types result from genetic variations in antigens on red blood cell surfaces, primarily the A, B, and Rh factor proteins.",
            "Q: Why is encryption important for digital security? A: Encryption transforms readable data into ciphertext using algorithms, ensuring confidentiality by making data unreadable without the correct decryption key.",
            "Q: What is the P versus NP problem in mathematics? A: P vs NP asks whether every problem whose solution can be quickly verified can also be quickly solved—a fundamental unsolved question in computer science.",
            "Q: How do neural networks learn from data? A: Neural networks learn by adjusting connection weights through backpropagation, minimizing prediction errors iteratively on training examples.",
            "Q: What causes tides in Earth's oceans? A: Tides result from gravitational forces exerted by the moon and sun, creating bulges in ocean water as Earth rotates.",
            "Q: Why is biodiversity important for ecosystems? A: Biodiversity provides ecosystem resilience, stability, and services through species interdependence, genetic variation, and functional redundancy.",
            "Q: What is the difference between bandwidth and latency? A: Bandwidth measures data transfer capacity (bits per second), while latency measures delay (time) in data transmission between two points.",
            "Q: How do black holes form? A: Black holes form when massive stars collapse under their own gravity after exhausting nuclear fuel, creating regions where gravity prevents even light from escaping.",
            "Q: What is the difference between syntax and semantics in programming? A: Syntax defines the grammatical rules of code structure, while semantics defines the meaning and behavior of syntactically correct code.",
            "Q: Why do leaves change color in autumn? A: Leaves change color as chlorophyll breaks down in cooler temperatures, revealing underlying carotenoids (yellows/oranges) and producing anthocyanins (reds).",
            "Q: What is cloud computing architecture? A: Cloud computing provides on-demand access to shared computing resources (servers, storage, applications) over the internet with scalable infrastructure.",
            "Q: How does photosynthesis convert light into chemical energy? A: Photosynthesis captures light energy to split water molecules and combine hydrogen with carbon dioxide, producing glucose and releasing oxygen.",
        ] * data_repetition_factor,
        eval_texts=[
            "Q: What is the difference between HTTP and HTTPS? A:",
            "Q: How does the human immune system recognize pathogens? A:",
            "Q: What causes the Doppler effect in sound waves? A:",
            "Q: Why is sorting data computationally important? A:",
            "Q: What is quantum entanglement? A:",
            "Q: How do compilers translate code into machine instructions? A:",
            "Q: What causes earthquakes along tectonic plate boundaries? A:",
            "Q: Why is version control important in software development? A:",
            "Q: What is the difference between virus and bacteria? A:",
            "Q: How does public key cryptography work? A:",
            "Q: What causes ocean currents to circulate globally? A:",
            "Q: Why is continuous integration valuable in development? A:",
            "Q: What is the Big Bang theory of universe origin? A:",
            "Q: How do search engines rank web pages? A:",
            "Q: What causes atmospheric pressure differences? A:",
        ],
    )

    return tasks
