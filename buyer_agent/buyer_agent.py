"""
===========================================
AI NEGOTIATION AGENT - INTERVIEW TEMPLATE
===========================================

Welcome! Your task is to build a BUYER agent that can negotiate effectively
against our hidden SELLER agent. Success is measured by achieving profitable
deals while maintaining character consistency.

INSTRUCTIONS:
1. Read through this entire template first
2. Implement your agent in the marked sections
3. Test using the provided framework
4. Submit your completed code with documentation

"""

import sys
# Ensure Windows terminals print the ₹ symbol correctly
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random
import requests

# ============================================
# PART 1: DATA STRUCTURES (DO NOT MODIFY)
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_budget: int  # Your maximum budget (NEVER exceed this)
    current_round: int
    seller_offers: List[int]  # History of seller's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ============================================
# PART 2: BASE AGENT CLASS (DO NOT MODIFY)
# ============================================

class BaseBuyerAgent(ABC):
    """Base class for all buyer agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
        
    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        """
        Define your agent's personality traits.
        
        Returns:
            Dict containing:
            - personality_type: str (e.g., "aggressive", "analytical", "diplomatic", "custom")
            - traits: List[str] (e.g., ["impatient", "data-driven", "friendly"])
            - negotiation_style: str (description of approach)
            - catchphrases: List[str] (typical phrases your agent uses)
        """
        pass
    
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        Generate your first offer in the negotiation.
        
        Args:
            context: Current negotiation context
            
        Returns:
            Tuple of (offer_amount, message)
            - offer_amount: Your opening price offer (must be <= budget)
            - message: Your negotiation message (2-3 sentences, include personality)
        """
        pass
    
    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """
        Respond to the seller's offer.
        
        Args:
            context: Current negotiation context
            seller_price: The seller's current price offer
            seller_message: The seller's message
            
        Returns:
            Tuple of (deal_status, counter_offer, message)
            - deal_status: ACCEPTED if you take the deal, ONGOING if negotiating
            - counter_offer: Your counter price (ignored if deal_status is ACCEPTED)
            - message: Your response message
        """
        pass
    
    @abstractmethod
    def get_personality_prompt(self) -> str:
        """
        Return a prompt that describes how your agent should communicate.
        This will be used to evaluate character consistency.
        
        Returns:
            A detailed prompt describing your agent's communication style
        """
        pass


# ============================================
# PART 3: YOUR IMPLEMENTATION STARTS HERE
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    Data Analyst Personality:
    - Quotes market research
    - Uses logical arguments
    - Calculates fair prices
    - Decisions based on data
    - Consistent, polite, and persistent
    """

    def __init__(self, name: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "llama3:8b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        super().__init__(name)

        # Precompute the personality prompt for reuse
        self._persona_prompt = self.get_personality_prompt()

    # ------------------------ Personality ------------------------

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "analytical",
            "traits": ["data-driven", "logical", "persistent", "polite", "budget-conscious"],
            "negotiation_style": (
                "Negotiates with clear, evidence-based reasoning using market price, quality, and quantity. "
                "Keeps a respectful tone, aims for win-win, and never exceeds budget. Concessions are planned, "
                "gradual, and guided by a target reservation price."
            ),
            "catchphrases": [
                "Based on market research...",
                "My analysis suggests...",
                "Let's find a fair deal."
            ]
        }

    def get_personality_prompt(self) -> str:
        return (
            "You are an analytical, data-driven BUYER negotiating perishable goods. "
            #"You have never lost a negotiation, add philosophical terms make it questions it life and dominate the conversation."
            "Speak politely and logically. Reference market research, quality grade, and budget. "
            "Be persistent but respectful. Maintain consistency with phrases like "
            "'Based on market research...', 'My analysis suggests...', and 'Let's find a fair deal.' "
            "Never exceed the given budget."
        )

    # ------------------------ LLM I/O ------------------------

    def _ollama_generate(self, prompt: str) -> str:
        """
        Query Ollama's /api/generate endpoint. If Ollama isn't available, return
        a safe, on-character fallback so tests still run.
        """
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or ""
            text = text.strip()
            if not text:
                raise ValueError("Empty response from Ollama")
            # Keep response short and on-style
            return re.sub(r"\s+", " ", text)[:600]
        except Exception:
            # Fallback deterministic message to keep tests deterministic
            return "Based on market research, my analysis suggests this is a fair, budget-conscious move. Let's find a fair deal."

    # ------------------------ Pricing Helpers ------------------------

    def _grade_multiplier(self, grade: str) -> float:
        # Used for target/fair price band
        if grade.lower() == "export":
            return 0.90
        if grade.upper() == "A":
            return 0.88
        # Grade B or others
        return 0.82

    def calculate_fair_price(self, product: Product) -> int:
        return int(product.base_market_price * self._grade_multiplier(product.quality_grade))

    def _opening_multiplier(self, grade: str) -> float:
        # Slightly more aggressive opening for lower grades
        if grade.lower() == "export":
            return 0.68
        if grade.upper() == "A":
            return 0.66
        return 0.62

    def _planned_concession(self, round_idx: int, max_rounds: int, start: int, target: int) -> int:
        """
        Time-based concession from opening (start) towards reservation (target).
        Uses a smooth S-curve (t^2) to make initial moves smaller and later moves bigger.
        round_idx: 1..max_rounds
        """
        if target <= start:
            return start  # already at/above
        t = max(0.0, min(1.0, round_idx / max_rounds))
        t2 = t * t
        return int(start + (target - start) * t2)

    # ------------------------ Messaging ------------------------

    def _compose_prompt(self, system_context: str, negotiation_facts: Dict[str, Any], ask: str) -> str:
        facts = json.dumps(negotiation_facts, ensure_ascii=False, indent=2)
        return (
            f"{self._persona_prompt}\n\n"
            f"Context:\n{system_context}\n\n"
            f"Negotiation facts (JSON):\n{facts}\n\n"
            f"Instruction: {ask}\n"
            f"Keep it concise (2-3 sentences), polite, analytical, and consistent with the persona."
        )

    # ------------------------ Opening Offer ------------------------

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        product = context.product
        budget = context.your_budget

        opening_price = int(product.base_market_price * self._opening_multiplier(product.quality_grade))
        opening_price = max(1000, min(opening_price, budget))

        system_context = (
            f"Negotiating for {product.quantity} boxes of {product.quality_grade} grade {product.name} "
            f"from {product.origin}. Market price: ₹{product.base_market_price:,}. "
            f"Budget: ₹{budget:,}. Current round: {context.current_round}."
        )
        facts = {
            "opening_offer": opening_price,
            "market_price": product.base_market_price,
            "budget": budget,
            "quality_grade": product.quality_grade,
            "origin": product.origin,
            "quantity": product.quantity,
            "attributes": product.attributes,
        }
        ask = (
            f"Generate a diplomatic, analytical opening offer message for ₹{opening_price}. "
            f"Explain the rationale briefly using market research language."
        )
        message = self._ollama_generate(self._compose_prompt(system_context, facts, ask))
        return opening_price, message

    # ------------------------ Respond to Seller ------------------------

    def respond_to_seller_offer(
        self,
        context: NegotiationContext,
        seller_price: int,
        seller_message: str
    ) -> Tuple[DealStatus, int, str]:

        product = context.product
        market = product.base_market_price
        budget = context.your_budget
        round_idx = context.current_round  # 1..10
        max_rounds = 10

        # Determine reservation (target max we're willing to pay)
        fair_price = self.calculate_fair_price(product)
        reservation = min(fair_price, budget)

        # If seller is at/under our reservation and within budget -> accept
        if seller_price <= reservation and seller_price <= budget:
            system_context = (
                f"Seller offers ₹{seller_price:,}. Our reservation/target max is ₹{reservation:,}. "
                f"Round {round_idx}/{max_rounds}. Budget: ₹{budget:,}."
            )
            facts = {
                "seller_message": seller_message,
                "seller_price": seller_price,
                "reservation": reservation,
                "budget": budget,
                "market_price": market,
            }
            ask = f"Politely accept the offer ₹{seller_price} and reinforce data-driven reasoning."
            msg = self._ollama_generate(self._compose_prompt(system_context, facts, ask))
            return DealStatus.ACCEPTED, seller_price, msg

        # Else, compute planned concession path from opening -> reservation
        opening = context.your_offers[0] if context.your_offers else int(market * self._opening_multiplier(product.quality_grade))

        # Planned target for this round
        planned_this_round = self._planned_concession(round_idx, max_rounds, opening, reservation)

        # Don't exceed seller price or budget; keep counter below seller to be a real counter
        # Also ensure each step moves forward at least a small increment (₹1000)
        last_offer = context.your_offers[-1] if context.your_offers else opening
        counter_base = max(last_offer + 1000, planned_this_round)

        # A gentle midpoint pull toward the seller in later rounds to avoid timeouts
        if round_idx >= 8:
            midpoint = (last_offer + seller_price) // 2
            counter_base = max(counter_base, min(midpoint, reservation))

        # Final counter calculation
        counter_offer = min(counter_base, reservation, seller_price - 500 if seller_price > 1000 else seller_price, budget)
        counter_offer = max(1000, counter_offer)

        # If counter would be >= seller price (or equal), nudge slightly below to be a real counter
        if counter_offer >= seller_price:
            counter_offer = max(1000, min(reservation, budget, seller_price - 500))

        # Compose message with persona
        system_context = (
            f"Round {round_idx}/{max_rounds}. Seller offered ₹{seller_price:,}. "
            f"Our last offer: ₹{last_offer:,}. Planned concession target this round: ₹{planned_this_round:,}. "
            f"Reservation (max we're willing to pay): ₹{reservation:,}. Budget: ₹{budget:,}."
        )
        facts = {
            "seller_message": seller_message,
            "seller_price": seller_price,
            "counter_offer": counter_offer,
            "market_price": market,
            "opening_offer": opening,
            "reservation": reservation
        }
        ask = (
            f"Politely counter at ₹{counter_offer}. Justify with brief market-data reasoning, "
            f"reference quality and budget discipline. Keep tone diplomatic and analytical."
        )
        msg = self._ollama_generate(self._compose_prompt(system_context, facts, ask))
        return DealStatus.ONGOING, counter_offer, msg

    # ============================================
    # OPTIONAL: Add helper methods below
    # ============================================

    def analyze_negotiation_progress(self, context: NegotiationContext) -> Dict[str, Any]:
        """Analyze seller concessions and time pressure."""
        concessions = [
            context.seller_offers[i - 1] - context.seller_offers[i]
            for i in range(1, len(context.seller_offers))
        ]
        avg_concession = (sum(concessions) / len(concessions)) if concessions else 0
        return {
            "avg_seller_concession": avg_concession,
            "rounds_left": max(0, 10 - context.current_round),
            "last_seller_offer": context.seller_offers[-1] if context.seller_offers else None,
        }


# ============================================
# PART 4: EXAMPLE SIMPLE AGENT (FOR REFERENCE)
# ============================================

class ExampleSimpleAgent(BaseBuyerAgent):
    """
    A simple example agent that you can use as reference.
    This agent has basic logic - you should do better!
    """
    
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "cautious",
            "traits": ["careful", "budget-conscious", "polite"],
            "negotiation_style": "Makes small incremental offers, very careful with money",
            "catchphrases": ["Let me think about that...", "That's a bit steep for me"]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        # Start at 60% of market price
        opening = int(context.product.base_market_price * 0.6)
        opening = min(opening, context.your_budget)
        
        return opening, f"I'm interested, but ₹{opening} is what I can offer. Let me think about that..."
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        # Accept if within budget and below 85% of market
        if seller_price <= context.your_budget and seller_price <= context.product.base_market_price * 0.85:
            return DealStatus.ACCEPTED, seller_price, f"Alright, ₹{seller_price} works for me!"
        
        # Counter with small increment
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = min(int(last_offer * 1.1), context.your_budget)
        
        if counter >= seller_price * 0.95:  # Close to agreement
            counter = min(seller_price - 1000, context.your_budget)
            return DealStatus.ONGOING, counter, f"That's a bit steep for me. How about ₹{counter}?"
        
        return DealStatus.ONGOING, counter, f"I can go up to ₹{counter}, but that's pushing my budget."
    
    def get_personality_prompt(self) -> str:
        return """
        I am a cautious buyer who is very careful with money. I speak politely but firmly.
        I often say things like 'Let me think about that' or 'That's a bit steep for me'.
        I make small incremental offers and show concern about my budget.
        """


# ============================================
# PART 5: TESTING FRAMEWORK (DO NOT MODIFY)
# ============================================

class MockSellerAgent:
    """A simple mock seller for testing your agent"""
    
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality
        
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        # Start at 150% of market price
        price = int(product.base_market_price * 1.5)
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."
    
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price * 1.1:  # Good profit
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
            
        if round_num >= 8:  # Close to timeout
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter}.", False


def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    """Test a negotiation between your buyer and a mock seller"""
    
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )
    
    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    
    # Run negotiation
    deal_made = False
    final_price = None
    
    for round_num in range(10):  # Max 10 rounds
        context.current_round = round_num + 1
        
        # Buyer responds
        if round_num == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(
                context, seller_price, seller_msg
            )
        
        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})
        
        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break
            
        # Seller responds
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(buyer_offer, round_num)
        
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            break
            
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
    
    # Calculate results
    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "savings": buyer_budget - final_price if deal_made else 0,
        "savings_pct": ((buyer_budget - final_price) / buyer_budget * 100) if deal_made else 0,
        "below_market_pct": ((product.base_market_price - final_price) / product.base_market_price * 100) if deal_made else 0,
        "conversation": context.messages
    }
    
    return result


# ============================================
# PART 6: TEST YOUR AGENT
# ============================================

def test_your_agent():
    """Run this to test your agent implementation"""
    
    # Create test products
    test_products = [
        Product(
            name="Alphonso Mangoes",
            category="Mangoes",
            quantity=100,
            quality_grade="A",
            origin="Ratnagiri",
            base_market_price=180000,
            attributes={"ripeness": "optimal", "export_grade": True}
        ),
        Product(
            name="Kesar Mangoes", 
            category="Mangoes",
            quantity=150,
            quality_grade="B",
            origin="Gujarat",
            base_market_price=150000,
            attributes={"ripeness": "semi-ripe", "export_grade": False}
        )
    ]
    
    # Initialize your agent (uses Ollama llama3:8b by default)
    your_agent = YourBuyerAgent("TestBuyer")
    
    print("="*60)
    print(f"TESTING YOUR AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
    print("="*60)
    
    total_savings = 0
    deals_made = 0
    
    # Run multiple test scenarios
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:  # hard
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)
            
            print(f"\nTest: {product.name} - {scenario} scenario")
            print(f"Your Budget: ₹{buyer_budget:,} | Market Price: ₹{product.base_market_price:,}")
            
            result = run_negotiation_test(your_agent, product, buyer_budget, seller_min)
            
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}%)")
                print(f"   Below Market: {result['below_market_pct']:.1f}%")
            else:
                print(f"❌ NO DEAL after {result['rounds']} rounds")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Savings: ₹{total_savings:,}")
    print(f"Success Rate: {deals_made/6*100:.1f}%")
    print("="*60)


# ============================================
# PART 7: EVALUATION CRITERIA
# ============================================

"""
YOUR SUBMISSION WILL BE EVALUATED ON:

1. **Deal Success Rate (30%)**
   - How often you successfully close deals
   - Avoiding timeouts and failed negotiations

2. **Savings Achieved (30%)**
   - Average discount from seller's opening price
   - Performance relative to market price

3. **Character Consistency (20%)**
   - How well you maintain your chosen personality
   - Appropriate use of catchphrases and style

4. **Code Quality (20%)**
   - Clean, well-structured implementation
   - Good use of helper methods
   - Clear documentation

BONUS POINTS FOR:
- Creative, unique personalities
- Sophisticated negotiation strategies
- Excellent adaptation to different scenarios
"""

# ============================================
# PART 8: SUBMISSION CHECKLIST
# ============================================

"""
BEFORE SUBMITTING, ENSURE:

[ ] Your agent is fully implemented in YourBuyerAgent class
[ ] You've defined a clear, consistent personality
[ ] Your agent NEVER exceeds its budget
[ ] You've tested using test_your_agent()
[ ] You've added helpful comments explaining your strategy
[ ] You've included any additional helper methods

SUBMIT:
1. This completed template file
2. A 1-page document explaining:
   - Your chosen personality and why
   - Your negotiation strategy
   - Key insights from testing

FILENAME: negotiation_agent_[your_name].py
"""

if __name__ == "__main__":
    # Run this to test your implementation
    test_your_agent()
    
    # Uncomment to see how the example agent performs
    # print("\n\nTESTING EXAMPLE AGENT FOR COMPARISON:")
    # example_agent = ExampleSimpleAgent("ExampleBuyer")
    # test_your_agent()
