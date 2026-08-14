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

# ---------- Configuración ----------
# Cada llave se lee justo cuando la función que la necesita se ejecuta, no al arrancar
# el script. Así el Paso 1 (generar) no falla por no tener las llaves que solo usa el Paso 2 (subir a D-ID).

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


def fetch_article_excerpt(url, max_chars=800):
    """Trae un extracto crudo del artículo para dar contexto a OpenAI (no es scraping fino,
    solo texto de la página con las etiquetas HTML quitadas)."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        text = re.sub("<[^<]+?>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception:
        return ""


def select_top_articles(client, articles):
    """Paso 1: con solo los títulos (barato, sin traer el contenido de cada artículo),
    le pedimos a OpenAI que elija entre 2 y 4 noticias relevantes y variadas.
    Así no gastamos de más trayendo el contenido completo de artículos que ni van a usarse."""
    candidates = articles[:MAX_ARTICLES_TO_OPENAI]
    titles_list = "\n".join(f"{i}: {a['title']}" for i, a in enumerate(candidates))

    prompt = f"""De esta lista de titulares recientes, elige entre 2 y 4 que sean realmente
relevantes y variados en tema (Perú, política, internacional, deportes, entretenimiento).
No elijas una noticia solo para llenar una categoría.

Responde ÚNICAMENTE con los números de índice elegidos, separados por coma (ejemplo: 0,3,7).
Sin texto adicional.

Titulares:
{titles_list}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content.strip()

    indices = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece.isdigit():
            idx = int(piece)
            if 0 <= idx < len(candidates):
                indices.append(idx)

    if not indices:
        # Si por algún motivo OpenAI no devolvió índices utilizables, no nos quedamos sin nada:
        # tomamos las primeras (más recientes) como respaldo.
        indices = list(range(min(4, len(candidates))))

    return [candidates[i] for i in indices]


def build_digest(articles):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    selected = select_top_articles(client, articles)

    # Paso 2: SOLO para las noticias ya elegidas, traemos más contenido del artículo original
    # (un extracto más largo que antes) para poder armar un desarrollo más completo de cada una.
    material = "\n\n---\n\n".join(
        f"Título: {a['title']}\nURL: {a['url']}\nContenido: {fetch_article_excerpt(a['url'], max_chars=2500)}"
        for a in selected
    )

    prompt = f"""Eres el editor detrás de Adrián, presentador virtual de Latina.
A partir de estas noticias ya seleccionadas, arma DOS secciones para que Adrián las use en
conversación con el público. Texto plano en ambas, sin markdown, sin viñetas, sin links -
va a ser leído en voz alta por un avatar.

SECCIÓN 1 - RESUMEN BREVE:
Un resumen conversacional de las {len(selected)} noticias juntas, para responder cuando alguien
pregunte "qué hay de nuevo" en general. 1-2 líneas por noticia.

SECCIÓN 2 - DETALLE POR NOTICIA:
Para cada noticia por separado, un desarrollo más completo (6-10 líneas) con más contexto,
datos concretos y por qué importa - para cuando alguien pregunte específicamente por ese tema.
Encabeza cada una con el título de la noticia tal cual para que quede claro a qué corresponde.

Noticias seleccionadas:
{material}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    digest_text = completion.choices[0].message.content.strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"NOTICIAS DE HOY (actualizado: {timestamp})\n\n{digest_text}\n"


def update_did_knowledge():
    did_api_key = os.environ["DID_API_KEY"]
    did_knowledge_id = os.environ["DID_KNOWLEDGE_ID"]
    digest_public_url = os.environ["DIGEST_PUBLIC_URL"]

    headers = {
        "Authorization": f"Basic {did_api_key}",
        "Content-Type": "application/json",
    }

    # 1. Borrar el/los documento(s) de noticias anteriores para que no queden
    #    versiones viejas conviviendo con la nueva (el riesgo que hablamos antes).
    docs_resp = requests.get(
        f"https://api.d-id.com/knowledge/{did_knowledge_id}/documents", headers=headers
    )
    docs_resp.raise_for_status()
    docs_data = docs_resp.json()
    # D-ID devuelve la lista de documentos directamente (no envuelta en {"documents": [...]})
    # pero dejamos el caso alternativo por si acaso, para no volver a romper aquí.
    docs_list = docs_data if isinstance(docs_data, list) else docs_data.get("documents", [])

    for doc in docs_list:
        if doc.get("title") == "Noticias del día":
            requests.delete(
                f"https://api.d-id.com/knowledge/{did_knowledge_id}/documents/{doc['id']}",
                headers=headers,
            )

    # 2. Registrar el nuevo documento, apuntando al digest.txt ya publicado en GitHub.
    create_resp = requests.post(
        f"https://api.d-id.com/knowledge/{did_knowledge_id}/documents",
        headers=headers,
        json={
            "title": "Noticias del día",
            "documentType": "text",
            "source_url": digest_public_url,
        },
    )
    if not create_resp.ok:
        # Imprimimos el cuerpo de la respuesta de D-ID para saber EXACTAMENTE qué campo
        # no le gustó, en vez de quedarnos solo con "400 Bad Request".
        print("Respuesta de D-ID al crear el documento:")
        print(create_resp.status_code, create_resp.text)
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
