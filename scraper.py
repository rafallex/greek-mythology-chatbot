"""
Greek Mythology Comprehensive Data Scraper - FIXED
Scrapes detailed articles and stories about Greek gods from Theoi.com
Uses correct URL structure from the actual site
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict
from urllib.parse import urljoin

class GreekGodsDataScraper:
    def __init__(self):
        self.base_url = "https://www.theoi.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.scraped_urls = set()
        self.all_gods_data = []
        
    def get_major_gods_urls(self) -> List[Dict[str, str]]:
        """Get URLs for major Greek gods - CORRECTED URLS"""
        major_gods = [
            {"name": "Zeus", "url": "/Olympios/Zeus.html"},
            {"name": "Hera", "url": "/Olympios/Hera.html"},
            {"name": "Poseidon", "url": "/Olympios/Poseidon.html"},
            {"name": "Demeter", "url": "/Olympios/Demeter.html"},
            {"name": "Athena", "url": "/Olympios/Athena.html"},
            {"name": "Apollo", "url": "/Olympios/Apollon.html"},
            {"name": "Artemis", "url": "/Olympios/Artemis.html"},
            {"name": "Ares", "url": "/Olympios/Ares.html"},
            {"name": "Aphrodite", "url": "/Olympios/Aphrodite.html"},
            {"name": "Hephaestus", "url": "/Olympios/Hephaistos.html"},
            {"name": "Hermes", "url": "/Olympios/Hermes.html"},
            {"name": "Dionysus", "url": "/Olympios/Dionysos.html"},
            {"name": "Hestia", "url": "/Olympios/Hestia.html"},
            
            # TITANS
            {"name": "Kronos", "url": "/Titan/TitanKronos.html"},
            {"name": "Rhea", "url": "/Titan/TitanisRhea.html"},
            {"name": "Oceanus", "url": "/Titan/TitanOkeanos.html"},
            {"name": "Tethys", "url": "/Titan/TitanisTethys.html"},
            {"name": "Hyperion", "url": "/Titan/TitanHyperion.html"},
            {"name": "Theia", "url": "/Titan/TitanisTheia.html"},
            {"name": "Coeus", "url": "/Titan/TitanKoios.html"},
            {"name": "Phoebe", "url": "/Titan/TitanisPhoibe.html"},
            {"name": "Iapetus", "url": "/Titan/TitanIapetos.html"},
            {"name": "Themis", "url": "/Titan/TitanisThemis.html"},
            {"name": "Mnemosyne", "url": "/Titan/TitanisMnemosyne.html"},
            {"name": "Crius", "url": "/Titan/TitanKrios.html"},
            {"name": "Prometheus", "url": "/Titan/TitanPrometheus.html"},
            {"name": "Atlas", "url": "/Titan/TitanAtlas.html"},
            {"name": "Epimetheus", "url": "/Titan/TitanEpimetheus.html"},
            {"name": "Menoetius", "url": "/Titan/TitanMenoitios.html"},
            {"name": "Helios", "url": "/Titan/Helios.html"},
            {"name": "Selene", "url": "/Titan/Selene.html"},
            {"name": "Eos", "url": "/Titan/Eos.html"},
            
            # PRIMORDIAL GODS
            {"name": "Chaos", "url": "/Protogenos/Khaos.html"},
            {"name": "Gaia", "url": "/Protogenos/Gaia.html"},
            {"name": "Uranus", "url": "/Protogenos/Ouranos.html"},
            {"name": "Nyx", "url": "/Protogenos/Nyx.html"},
            {"name": "Erebus", "url": "/Protogenos/Erebos.html"},
            {"name": "Tartarus", "url": "/Protogenos/Tartaros.html"},
            {"name": "Eros", "url": "/Protogenos/Eros.html"},
            {"name": "Aether", "url": "/Protogenos/Aither.html"},
            {"name": "Hemera", "url": "/Protogenos/Hemera.html"},
            
            # UNDERWORLD GODS
            {"name": "Hades", "url": "/Khthonios/Haides.html"},
            {"name": "Persephone", "url": "/Khthonios/Persephone.html"},
            {"name": "Thanatos", "url": "/Daimon/Thanatos.html"},
            {"name": "Hypnos", "url": "/Daimon/Hypnos.html"},
            {"name": "Charon", "url": "/Khthonios/Kharon.html"},
            {"name": "Hecate", "url": "/Khthonios/Hekate.html"},
            {"name": "Erinyes (Furies)", "url": "/Khthonios/Erinyes.html"},
            
            # SEA GODS
            {"name": "Nereus", "url": "/Pontios/Nereus.html"},
            {"name": "Proteus", "url": "/Pontios/Proteus.html"},
            {"name": "Triton", "url": "/Pontios/Triton.html"},
            {"name": "Amphitrite", "url": "/Pontios/Amphitrite.html"},
            
            # SKY/NATURE GODS
            {"name": "Pan", "url": "/Georgikos/Pan.html"},
            {"name": "Iris", "url": "/Pontios/Iris.html"},
            
            # ABSTRACT/PERSONIFICATION GODS
            {"name": "Nike", "url": "/Daimon/Nike.html"},
            {"name": "Tyche", "url": "/Daimon/Tykhe.html"},
            {"name": "Nemesis", "url": "/Daimon/Nemesis.html"},
            {"name": "Eris", "url": "/Daimon/Eris.html"},
            {"name": "Moirai (Fates)", "url": "/Daimon/Moirai.html"},
            
            # MUSES AND GRACES
            {"name": "The Muses", "url": "/Ouranios/Mousai.html"},
            {"name": "The Charites (Graces)", "url": "/Ouranios/Kharites.html"},
            
            # HEROIC/DEIFIED
            {"name": "Heracles", "url": "/Heros/Herakles.html"},
            {"name": "Asclepius", "url": "/Ouranios/Asklepios.html"},
        ]
        
        return [{"name": g["name"], "url": urljoin(self.base_url, g["url"])} for g in major_gods]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?()\-\'"–—]+', '', text) 
        return text.strip()
    
    def extract_article_content(self, soup) -> str:
        """Extract main article content"""
        for unwanted in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form', 'input']):
            unwanted.decompose()
        
        for nav_elem in soup.find_all(['div', 'aside'], class_=re.compile(r'nav|menu|sidebar|footer', re.I)):
            nav_elem.decompose()
        
        texts = []
        
        for elem in soup.find_all(['p', 'blockquote', 'div']):
            text = elem.get_text(separator=' ', strip=True)
            if (len(text) > 50 and 
                not re.search(r'copyright|rights reserved|click here|home \||back to', text, re.I)):
                texts.append(text)
        
        full_text = '\n\n'.join(texts)
        
        paragraphs = full_text.split('\n\n')
        unique_paragraphs = []
        seen = set()
        for p in paragraphs:
            p_clean = p.strip()
            if p_clean and p_clean not in seen and len(p_clean) > 50:
                unique_paragraphs.append(p_clean)
                seen.add(p_clean)
        
        return '\n\n'.join(unique_paragraphs)
    
    def scrape_god_page(self, god_info: Dict[str, str]) -> Dict:
        """Scrape a single god's page"""
        url = god_info['url']
        
        if url in self.scraped_urls:
            return None
        
        self.scraped_urls.add(url)
        
        try:
            print(f"Scraping: {god_info['name']}...")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_content = self.extract_article_content(soup)
            
            if len(article_content) < 200:
                print(f"  ⚠ Limited content for {god_info['name']} ({len(article_content)} chars)")
                if len(article_content) < 100:
                    return None
            
            god_data = {
                "god_name": god_info['name'],
                "source_url": url,
                "content": self.clean_text(article_content),
                "content_length": len(article_content),
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"  ✓ Success: {len(article_content):,} characters")
            return god_data
            
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP Error for {god_info['name']}: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error for {god_info['name']}: {e}")
            return None
    
    def scrape_all_gods(self) -> List[Dict]:
        """Scrape all major gods"""
        gods_list = self.get_major_gods_urls()
        print(f"\n{'='*70}")
        print(f"Starting to scrape {len(gods_list)} Greek gods from Theoi.com")
        print(f"{'='*70}\n")
        
        successful = 0
        failed = 0
        
        for i, god_info in enumerate(gods_list, 1):
            print(f"[{i}/{len(gods_list)}] ", end="")
            
            god_data = self.scrape_god_page(god_info)
            
            if god_data:
                self.all_gods_data.append(god_data)
                successful += 1
            else:
                failed += 1
            
            # Rate limiting
            time.sleep(1.2)
            
            if i % 10 == 0:
                print(f"\n{'─'*70}")
                print(f"Checkpoint: {successful} successful, {failed} failed")
                print(f"{'─'*70}\n")
                time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"SCRAPING COMPLETE!")
        print(f"Successfully scraped: {successful}/{len(gods_list)} gods")
        print(f"Failed: {failed}/{len(gods_list)} gods")
        print(f"{'='*70}\n")
        
        return self.all_gods_data
    
    def save_to_file(self, filename: str = "greek_gods_complete_data.json"):
        """Save all data to JSON"""
        sorted_data = sorted(self.all_gods_data, key=lambda x: x['content_length'], reverse=True)
        
        output = {
            "metadata": {
                "total_gods": len(sorted_data),
                "total_characters": sum(g['content_length'] for g in sorted_data),
                "average_characters_per_god": sum(g['content_length'] for g in sorted_data) // len(sorted_data) if sorted_data else 0,
                "source": "theoi.com",
                "scraped_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gods_list": [g['god_name'] for g in sorted_data]
            },
            "gods_data": sorted_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Data saved to: {filename}")
        print(f"  Total size: {sum(g['content_length'] for g in sorted_data):,} characters\n")
        
        return output
    
    def print_summary(self):
        """Print detailed summary"""
        if not self.all_gods_data:
            print("⚠ No data scraped!")
            return
        
        print(f"\n{'='*70}")
        print("DATASET SUMMARY")
        print(f"{'='*70}\n")
        
        total_chars = sum(g['content_length'] for g in self.all_gods_data)
        avg_chars = total_chars // len(self.all_gods_data)
        
        print(f"Total Gods: {len(self.all_gods_data)}")
        print(f"Total Content: {total_chars:,} characters (~{total_chars//1000}K)")
        print(f"Average per God: {avg_chars:,} characters")
        
        categories = {
            "OLYMPIAN GODS": ["Zeus", "Hera", "Poseidon", "Demeter", "Athena", "Apollo", 
                             "Artemis", "Ares", "Aphrodite", "Hephaestus", "Hermes", "Dionysus", "Hestia"],
            "TITANS": ["Kronos", "Rhea", "Oceanus", "Tethys", "Hyperion", "Theia", "Coeus", 
                      "Phoebe", "Iapetus", "Themis", "Mnemosyne", "Crius", "Prometheus", 
                      "Atlas", "Epimetheus", "Menoetius", "Helios", "Selene", "Eos"],
            "PRIMORDIAL DEITIES": ["Chaos", "Gaia", "Uranus", "Nyx", "Erebus", "Tartarus", 
                                   "Eros", "Aether", "Hemera"],
            "UNDERWORLD GODS": ["Hades", "Persephone", "Thanatos", "Hypnos", "Charon", 
                               "Hecate", "Erinyes (Furies)"],
            "OTHER DEITIES": None 
        }
        
        god_names = {g['god_name']: g['content_length'] for g in self.all_gods_data}
        
        print(f"\n{'─'*70}")
        print("GODS INCLUDED IN DATASET (by category):")
        print(f"{'─'*70}")
        
        all_categorized = []
        for category, gods_list in categories.items():
            if gods_list is None:
                continue
            all_categorized.extend(gods_list)
            found = [g for g in gods_list if g in god_names]
            
            if found:
                print(f"\n{category}:")
                for god in found:
                    print(f"  • {god:30} {god_names[god]:>8,} chars")
        
        other_gods = [g for g in god_names.keys() if g not in all_categorized]
        if other_gods:
            print(f"\nOTHER DEITIES:")
            for god in other_gods:
                print(f"  • {god:30} {god_names[god]:>8,} chars")
        
        print(f"\n{'='*70}\n")


def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("GREEK GODS DATA SCRAPER v2.0")
    print("Comprehensive articles for LLM fine-tuning")
    print(f"{'='*70}")
    
    scraper = GreekGodsDataScraper()
    scraper.scrape_all_gods()
    scraper.save_to_file("greek_gods_complete_data.json")
    scraper.print_summary()
    
    print("✓ COMPLETE! Process 'greek_gods_complete_data.json' for fine-tuning.\n")


if __name__ == "__main__":
    main()