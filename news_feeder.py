"""
Adrián News Feeder
-------------------
1. Lee el sitemap de noticias de latinanoticias.pe (formato estándar Google News Sitemap).
2. Filtra los artículos publicados en las últimas HOURS_WINDOW horas.
3. Usa OpenAI para condensarlos en un "digest" breve, en el mismo formato/tono
   que ya usamos en el prompt de Adrián (2-4 noticias, variedad temática, sin markdown).
4. Guarda ese digest como digest.txt (para que GitHub Actions lo suba al repo).
5. Actualiza el Knowledge base de Adrián en D-ID: borra el documento de noticias
   anterior y registra el nuevo, apuntando a la URL pública del digest.txt en GitHub.

IMPORTANTE - cosas que no pude probar en vivo (sin acceso a internet desde donde escribo esto)
y que conviene revisar la primera vez que lo corras:
  - El formato exacto de las fechas dentro del sitemap (ajusté el parseo al estándar,
    pero si falla, imprime el XML crudo para revisar el formato real).
  - El endpoint de borrar un documento (DELETE /knowledge/{id}/documents/{docId}) lo inferí
    del patrón REST de D-ID, pero no está explícitamente documentado en lo que revisé -
    si da error, revisa la sección "Documents" de la API Reference de D-ID.
  - El origen de la publicación en raw.githubusercontent.com puede tardar unos minutos en
    reflejar el último commit (cache de GitHub) - no te alarmes si D-ID no ve el cambio al instante.
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from openai import OpenAI

# ---------- Configuración (se leen como variables de entorno / GitHub Secrets) ----------
DID_API_KEY = os.environ["DID_API_KEY"]              # tu API key de D-ID (Basic auth)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]         # tu API key de OpenAI
DID_KNOWLEDGE_ID = os.environ["DID_KNOWLEDGE_ID"]     # id del knowledge base de Adrián (knl_xxx)
DIGEST_PUBLIC_URL = os.environ["DIGEST_PUBLIC_URL"]   # URL pública donde quedará digest.txt tras el commit

SITEMAP_URL = "https://latinanoticias.pe/_files/sitemaps/sitemap_news.xml"
HOURS_WINDOW = 6          # solo considerar artículos de las últimas N horas
MAX_ARTICLES_TO_OPENAI = 15  # tope para no mandar de más y controlar costo

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


def fetch_recent_articles():
    resp = requests.get(SITEMAP_URL, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)
    articles = []

    for url_el in root.findall("sm:url", NS):
        loc = url_el.find("sm:loc", NS)
        news_el = url_el.find("news:news", NS)
        if loc is None or news_el is None:
            continue

        title_el = news_el.find("news:title", NS)
        date_el = news_el.find("news:publication_date", NS)
        if title_el is None or date_el is None:
            continue

        try:
            pub_date = datetime.fromisoformat(date_el.text.replace("Z", "+00:00"))
        except ValueError:
            # Si el formato de fecha viene distinto al esperado, lo saltamos
            # en vez de tumbar todo el script.
            continue

        if pub_date < cutoff:
            continue

        articles.append({"url": loc.text, "title": title_el.text, "date": pub_date})

    return articles


def fetch_article_excerpt(url):
    """Trae un extracto crudo del artículo para dar contexto a OpenAI (no es scraping fino,
    solo texto de la página con las etiquetas HTML quitadas)."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        text = re.sub("<[^<]+?>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
        return text[:800]
    except Exception:
        return ""


def build_digest(articles):
    client = OpenAI(api_key=OPENAI_API_KEY)

    raw_material = "\n\n".join(
        f"Título: {a['title']}\nURL: {a['url']}\nContexto: {fetch_article_excerpt(a['url'])}"
        for a in articles[:MAX_ARTICLES_TO_OPENAI]
    )

    prompt = f"""Eres el editor detrás de Adrián, presentador virtual de Latina.
A partir de estos titulares recientes, arma un digest breve para que Adrián lo use como base
de conversación con el público.

Reglas:
- Elige entre 2 y 4 noticias realmente relevantes, con variedad temática cuando exista
  (Perú, política, internacional, deportes, entretenimiento). No incluyas una noticia
  solo para llenar una categoría.
- Para cada una: un título corto + 1-2 líneas explicando qué pasó y por qué es relevante.
- Texto plano, sin markdown, sin viñetas, sin links - va a ser leído en voz alta por un avatar.

Material en bruto:
{raw_material}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    digest_text = completion.choices[0].message.content.strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"NOTICIAS DE HOY (actualizado: {timestamp})\n\n{digest_text}\n"


def update_did_knowledge():
    headers = {
        "Authorization": f"Basic {DID_API_KEY}",
        "Content-Type": "application/json",
    }

    # 1. Borrar el/los documento(s) de noticias anteriores para que no queden
    #    versiones viejas conviviendo con la nueva (el riesgo que hablamos antes).
    docs_resp = requests.get(
        f"https://api.d-id.com/knowledge/{DID_KNOWLEDGE_ID}/documents", headers=headers
    )
    docs_resp.raise_for_status()
    for doc in docs_resp.json().get("documents", []):
        if doc.get("title") == "Noticias del día":
            requests.delete(
                f"https://api.d-id.com/knowledge/{DID_KNOWLEDGE_ID}/documents/{doc['id']}",
                headers=headers,
            )

    # 2. Registrar el nuevo documento, apuntando al digest.txt ya publicado en GitHub.
    create_resp = requests.post(
        f"https://api.d-id.com/knowledge/{DID_KNOWLEDGE_ID}/documents",
        headers=headers,
        json={
            "title": "Noticias del día",
            "documentType": "txt",
            "source_url": DIGEST_PUBLIC_URL,
        },
    )
    create_resp.raise_for_status()


def step_generate():
    """Paso 1: genera digest.txt localmente. GitHub Actions debe hacer commit + push
    de este archivo antes de correr el paso 2, para que la URL pública ya sirva el contenido nuevo."""
    articles = fetch_recent_articles()

    if not articles:
        print("No se encontraron artículos nuevos en la ventana de tiempo definida.")
        # Igual dejamos un digest.txt vacío-informativo para no romper el paso siguiente.
        with open("digest.txt", "w", encoding="utf-8") as f:
            f.write("Sin actualizaciones recientes.\n")
        return

    digest = build_digest(articles)
    with open("digest.txt", "w", encoding="utf-8") as f:
        f.write(digest)

    print("digest.txt generado:\n")
    print(digest)


def step_update():
    """Paso 2: le dice a D-ID que borre el documento de noticias anterior y registre
    el nuevo, apuntando a la URL pública del digest.txt ya publicado en GitHub."""
    update_did_knowledge()
    print("Knowledge base de D-ID actualizado.")


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "generate":
        step_generate()
    elif step == "update":
        step_update()
    else:
        print("Uso: python news_feeder.py [generate|update]")
