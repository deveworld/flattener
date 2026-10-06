// Interface language: Korean when the browser's first preferred language is Korean,
// English otherwise. English text in the page and in the code is the source; `localize`
// translates the static page once and then every text or label the app writes.
import { ko, koPatterns } from './i18n/ko';

export type Locale = 'en' | 'ko';

function preferred(): Locale {
  const first = (globalThis.navigator?.languages?.[0] ?? globalThis.navigator?.language ?? '')
    .toLowerCase()
    .trim();
  return first === 'ko' || first.startsWith('ko-') ? 'ko' : 'en';
}

export const locale: Locale = preferred();

/** ``text`` in the interface language; unknown text stays as it is. */
export function translate(text: string): string {
  if (locale === 'en' || !text) return text;
  const exact = ko[text];
  if (exact !== undefined) return exact;
  for (const [pattern, replacement] of koPatterns)
    if (pattern.test(text)) return text.replace(pattern, replacement);
  // Messages built from several known sentences, such as joined warnings.
  const sentences = text.split(/(?<=[.!?])\s+/);
  if (sentences.length > 1) {
    const translated = sentences.map((sentence) => ko[sentence] ?? sentence);
    if (translated.some((sentence, index) => sentence !== sentences[index]))
      return translated.join(' ');
  }
  return text;
}

const ATTRIBUTES = ['aria-label', 'title', 'alt', 'placeholder'];

function kept(node: Node): boolean {
  const element = node instanceof Element ? node : node.parentElement;
  return !!element?.closest('[translate="no"], script, style');
}

function translateText(node: Node) {
  const value = node.nodeValue;
  if (!value || kept(node)) return;
  const text = value.trim();
  const translated = translate(text);
  if (translated !== text) node.nodeValue = value.replace(text, translated);
}

function translateAttribute(element: Element, name: string) {
  const value = element.getAttribute(name);
  if (!value || kept(element)) return;
  const translated = translate(value);
  if (translated !== value) element.setAttribute(name, translated);
}

function translateTree(root: Node) {
  if (root.nodeType === Node.TEXT_NODE) return translateText(root);
  if (!(root instanceof Element)) return;
  for (const name of ATTRIBUTES) translateAttribute(root, name);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeType === Node.TEXT_NODE) translateText(node);
    else for (const name of ATTRIBUTES) translateAttribute(node as Element, name);
  }
}

/** Translate the page now and keep translating what the app writes into it. */
export function localize() {
  document.documentElement.lang = locale;
  if (locale === 'en') return;
  document.title = translate(document.title);
  const description = document.querySelector('meta[name="description"]');
  if (description) translateAttribute(description, 'content');
  translateTree(document.body);
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'childList') record.addedNodes.forEach(translateTree);
      else if (record.type === 'characterData') translateText(record.target);
      else if (record.attributeName)
        translateAttribute(record.target as Element, record.attributeName);
    }
  }).observe(document.body, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: ATTRIBUTES,
  });
}
