# Jarvis V2 – Assistant Vocal Cyberpunk

Jarvis V2 est un assistant personnel intelligent, ultra-rapide et proactif, conçu pour fonctionner localement sous Windows 11 avec une accélération GPU (NVIDIA RTX). Il dispose d'une interface HUD style Cyberpunk/Iron Man et supporte l'interruption vocale (barge-in).

## 🚀 Architecture

L'architecture est entièrement asynchrone pour permettre une interaction fluide :

- **STT (Speech-To-Text) :** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (modèle medium) couplé à **Silero VAD** pour une détection précise de la voix.
- **LLM (Large Language Model) :** [Ollama](https://ollama.com/) (par défaut `llama3:8b`) fonctionnant en local.
- **TTS (Text-To-Speech) :** [edge-tts](https://github.com/rany2/edge-tts) pour des voix françaises naturelles, avec décodage via **pydub/ffmpeg**.
- **UI :** Interface graphique interactive développée avec **PyQt6**, simulant un "Arc Reactor" avec des effets de transparence et d'animations néon.

## ✨ Fonctionnalités clés

- **Interruption (Barge-in) :** Jarvis s'arrête immédiatement de parler dès qu'il détecte que vous reprenez la parole.
- **Mémoire Persistante :** Utilise SQLite pour mémoriser des faits sur l'utilisateur, ses préférences et ses projets.
- **Réponses en Streaming :** Le texte est généré et lu phrase par phrase pour réduire la latence.
- **Commandes Intégrées :** Le LLM peut déclencher des actions locales (ouvrir des apps, lancer de la musique, recherches web).
- **Interface Immersive :** Un HUD Cyberpunk qui réagit visuellement selon que Jarvis écoute, réfléchit ou parle.

## 📋 Prérequis

- **Système :** Windows 11 (recommandé).
- **Matériel :** GPU NVIDIA (ex: RTX 3070 Ti) avec drivers CUDA installés.
- **Logiciels :**
  - [Python 3.10+](https://www.python.org/)
  - [Ollama](https://ollama.com/) (avec le modèle `llama3:8b` installé)
  - [FFmpeg](https://ffmpeg.org/) (nécessaire pour le traitement audio de `pydub`)

## 🛠️ Installation

1. **Cloner le dépôt :**
   ```bash
   git clone <url-du-repo>
   cd jarvis-v2
   ```

2. **Installer les dépendances :**
   Il est recommandé d'utiliser un environnement virtuel.
   ```bash
   pip install -r requirements.txt
   ```

3. **Installer les modèles Ollama :**
   ```bash
   ollama pull llama3:8b
   ```

4. **Vérifier FFmpeg :**
   Assurez-vous que `ffmpeg` est dans votre PATH système.

## 🖥️ Utilisation

Lancez l'assistant avec la commande suivante :

```bash
python main.py
```

L'interface apparaîtra au centre de votre écran. Parlez naturellement en français.

## 🤖 Commandes supportées

Jarvis peut interpréter et exécuter les commandes suivantes dynamiquement :

- **Applications :** `[CMD: OPEN_APP('spotify')]`
- **Recherche Web :** `[CMD: SEARCH_WEB('météo Issoire')]`
- **Musique :** `[CMD: PLAY_MUSIC('playlist triste')]` (Supporte Spotify, YouTube et la musique locale dans `C:\musique`)
- **Mémoire :**
  - `[CMD: SAVE_FACT('nom_utilisateur', 'Fusion')]`
  - `[CMD: DELETE_FACT('nom_utilisateur')]`
- **Contrôle :** `[CMD: MIDI('commande')]` (Placeholder)

## 🧠 Configuration

Les paramètres principaux (modèles, seuils VAD, voix TTS) sont modifiables directement au début du fichier `main.py`.

---
*Développé pour une expérience fluide et immersive sous Windows 11.*
