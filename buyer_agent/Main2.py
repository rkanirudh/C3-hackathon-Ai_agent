"""
===========================================
AI NEGOTIATION AGENT - DUAL ROLE AGENT (FIXED)
===========================================

A versatile agent that automatically detects opponent's role and switches
between buyer/seller modes. This file contains a corrected implementation
and a test harness to exercise the agent against a mock opponent.
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
# DATA STRUCTURES
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
    your_role: str  # 'buyer' or 'seller' (detected automatically or preset)
    your_limit: int  # Budget for buyer, min_price for seller
    current_round: int
    opponent_offers: List[int]  # History of opponent's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ============================================
# BASE AGENT CLASS
# ============================================

class BaseNegotiationAgent(ABC):
    """Base class for dual-role agents"""

    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        """Define your agent's personality traits for both roles."""
        pass

    @abstractmethod
    def detect_opponent_role(self, context: NegotiationContext, opponent_message: str) -> str:
        """Detect if opponent is buyer or seller based on their message."""
        pass

    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate your first offer in the negotiation."""
        pass

    @abstractmethod
    def respond_to_offer(self, context: NegotiationContext, opponent_price: int, opponent_message: str) -> Tuple[DealStatus, int, str]:
        """Respond to the opponent's offer."""
        pass

# ============================================
# YOUR DUAL ROLE AGENT IMPLEMENTATION
# ============================================

class YourDualRoleAgent(BaseNegotiationAgent):
    """
    Smart Dual-Role Agent:
    - Automatically detects opponent's role (if desired)
    - Switches between buyer/seller strategies
    - Maintains consistent analytical personality
    """

    def __init__(self, name: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "llama3:8b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        super().__init__(name)
        self._buyer_prompt = self.get_personality_prompt("buyer")
        self._seller_prompt = self.get_personality_prompt("seller")

    def define_personality(self) -> Dict[str, Any]:
        return {
            "core_traits": ["analytical", "adaptive", "strategic", "professional"],
            "buyer_style": "Data-driven buyer focusing on market research and budget discipline",
            "seller_style": "Quality-focused seller emphasizing value and fair pricing",
            "catchphrases": [
                "Based on my analysis...",
                "Considering current market conditions...",
                "Let's find a mutually beneficial solution...",
                "The data suggests a fair value would be..."
            ]
        }

    def get_personality_prompt(self, role: str) -> str:
        if role == "buyer":
            return (
                "You are an analytical BUYER negotiating perishable goods. "
                "Use market data, quality assessment, and budget constraints. "
                "Be polite but firm. Reference research and aim for win-win outcomes."
            )
        else:
            return (
                "You are a professional SELLER emphasizing product quality and market value. "
                "Be confident but reasonable. Highlight unique features and aim for fair pricing."
            )

    def detect_opponent_role(self, context: NegotiationContext, opponent_message: str) -> str:
        """
        Detect opponent's role based on their message content and pricing behavior.
        If opponent sounds like a seller, we should act as buyer and vice versa.
        """
        message_lower = (opponent_message or "").lower()

        # Keyword-based detection
        seller_keywords = ['selling', 'price is', 'cost', 'value', 'premium', 'quality', 'asking', 'i\'m selling', 'i am selling']
        buyer_keywords = ['buying', 'budget', 'afford', 'expensive', 'counter', 'counteroffer', 'interested', 'i want to buy']

        seller_score = sum(1 for word in seller_keywords if word in message_lower)
        buyer_score = sum(1 for word in buyer_keywords if word in message_lower)

        # Price-based detection (if we have opponent offers)
        if context.opponent_offers:
            last_offer = context.opponent_offers[-1]
            market_price = context.product.base_market_price

            if last_offer > market_price * 1.2:  # Likely seller
                seller_score += 2
            elif last_offer < market_price * 0.8:  # Likely buyer
                buyer_score += 2

        # If opponent is more seller-like, our role should be buyer
        if seller_score > buyer_score:
            return "buyer"
        else:
            return "seller"

    def _ollama_generate(self, prompt: str) -> str:
        """Generate response using Ollama with fallback"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()[:500]
        except Exception:
            # Fallback short message
            return "Based on my analysis, I believe this represents fair market value."

    # ------------------------ PRICING STRATEGIES ------------------------

    def _buyer_grade_multiplier(self, grade: str) -> float:
        if grade.lower() == "export": return 0.90
        if grade.upper() == "A": return 0.88
        return 0.82

    def _seller_grade_multiplier(self, grade: str) -> float:
        if grade.lower() == "export": return 1.4
        if grade.upper() == "A": return 1.3
        return 1.2

    def _buyer_opening_multiplier(self, grade: str) -> float:
        if grade.lower() == "export": return 0.68
        if grade.upper() == "A": return 0.66
        return 0.62

    def _seller_opening_multiplier(self, grade: str) -> float:
        if grade.lower() == "export": return 1.5
        if grade.upper() == "A": return 1.4
        return 1.3

    def calculate_target_price(self, product: Product, role: str) -> int:
        if role == "buyer":
            return int(product.base_market_price * self._buyer_grade_multiplier(product.quality_grade))
        else:
            return int(product.base_market_price * self._seller_grade_multiplier(product.quality_grade))

    def _concession_strategy(self, round_idx: int, max_rounds: int, current: int, target: int, role: str) -> int:
        """Smart concession strategy for both roles (returns integer)"""
        progress = max(0.0, min(1.0, round_idx / max_rounds))

        if role == "buyer":
            if target <= current:
                return int(current)
            concession_rate = 0.2 + (progress * 0.6)  # 20-80% concession based on progress
            return int(min(target, current + int((target - current) * concession_rate)))
        else:
            if target >= current:
                return int(current)
            concession_rate = 0.1 + (progress * 0.5)  # 10-60% concession
            return int(max(target, current - int((current - target) * concession_rate)))

    # ------------------------ CORE NEGOTIATION ------------------------

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate opening offer based on detected role"""
        product = context.product
        role = context.your_role
        limit = context.your_limit

        if role == "buyer":
            multiplier = self._buyer_opening_multiplier(product.quality_grade)
            offer = int(product.base_market_price * multiplier)
            offer = min(offer, limit)
            message = f"Based on market research, I'm offering ₹{offer:,} for this {product.quality_grade} grade {product.name}."
        else:
            multiplier = self._seller_opening_multiplier(product.quality_grade)
            offer = int(product.base_market_price * multiplier)
            offer = max(offer, limit)
            message = f"This premium {product.quality_grade} grade {product.name} is valued at ₹{offer:,} based on current market rates."

        return int(offer), message

    def respond_to_offer(self, context: NegotiationContext, opponent_price: int, opponent_message: str) -> Tuple[DealStatus, int, str]:
        """Main response handler - automatically adapts to opponent's role"""
        product = context.product
        your_role = context.your_role
        your_limit = context.your_limit
        round_idx = context.current_round

        # Auto-detect role if first round or role not set
        if round_idx == 1 or not your_role:
            detected_role = self.detect_opponent_role(context, opponent_message)
            context.your_role = detected_role
            your_role = detected_role

        if your_role == "buyer":
            return self._respond_as_buyer(context, int(opponent_price), opponent_message)
        else:
            return self._respond_as_seller(context, int(opponent_price), opponent_message)

    def _respond_as_buyer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """Buyer response logic"""
        product = context.product
        budget = context.your_limit
        round_idx = context.current_round

        target_price = self.calculate_target_price(product, "buyer")
        reservation_price = min(target_price, budget)

        # Acceptance conditions
        if seller_price <= reservation_price:
            return DealStatus.ACCEPTED, int(seller_price), f"Accepted at ₹{int(seller_price):,}. This aligns with my market analysis."

        if round_idx >= 8 and seller_price <= budget:  # Late round concession
            return DealStatus.ACCEPTED, int(seller_price), f"I accept ₹{int(seller_price):,} to close the deal."

        # Calculate counter offer
        opening = int(context.your_offers[0]) if context.your_offers else int(self.calculate_target_price(product, "buyer") * 0.9)
        last_offer = int(context.your_offers[-1]) if context.your_offers else opening
        counter_offer = self._concession_strategy(round_idx, 10, last_offer, reservation_price, "buyer")
        counter_offer = min(int(counter_offer), reservation_price, seller_price - 1000)
        counter_offer = max(1000, int(counter_offer))

        message = f"Based on current market data, I can offer ₹{int(counter_offer):,}. This reflects the {product.quality_grade} grade quality."
        return DealStatus.ONGOING, int(counter_offer), message

    def _respond_as_seller(self, context: NegotiationContext, buyer_price: int, buyer_message: str) -> Tuple[DealStatus, int, str]:
        """Seller response logic"""
        product = context.product
        min_price = context.your_limit
        round_idx = context.current_round

        target_price = self.calculate_target_price(product, "seller")

        # Acceptance conditions (seller may accept above thresholds)
        if buyer_price >= min_price:
            if buyer_price >= min_price * 1.15 or (round_idx >= 8 and buyer_price >= min_price):
                return DealStatus.ACCEPTED, int(buyer_price), f"Accepted at ₹{int(buyer_price):,}. Thank you for your offer."

        # Calculate counter offer: start from last offer (or target) and concede toward min_price
        last_offer = int(context.your_offers[-1]) if context.your_offers else int(target_price)
        counter_offer = self._concession_strategy(round_idx, 10, last_offer, min_price, "seller")
        # Ensure counter_offer is at least min_price and slightly above buyer_price to encourage movement
        counter_offer = max(int(counter_offer), int(min_price), int(buyer_price) + 1000)

        message = f"This {product.quality_grade} grade {product.name} from {product.origin} is worth ₹{int(counter_offer):,} given its premium quality."
        return DealStatus.ONGOING, int(counter_offer), message

# ============================================
# TESTING FRAMEWORK
# ============================================

class MockOpponentAgent:
    """Mock opponent for testing - can be buyer or seller"""

    def __init__(self, role: str, limit: int):
        self.role = role
        self.limit = int(limit)

    def get_opening_offer(self, product: Product) -> Tuple[int, str]:
        if self.role == "seller":
            price = int(product.base_market_price * 1.5)
            return price, f"I'm selling premium {product.name} for ₹{price:,}"
        else:
            price = int(product.base_market_price * 0.6)
            return min(price, self.limit), f"I'm interested in buying at ₹{price:,}"

    def respond_to_offer(self, your_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if self.role == "seller":
            if your_offer >= int(self.limit * 1.1):
                return your_offer, "Deal!", True
            counter = max(self.limit, int(your_offer * 1.1))
            return counter, f"I can do ₹{counter:,}", False
        else:
            # Buyer: accepts if offer below threshold, otherwise counters
            if your_offer <= int(self.limit * 0.9):
                return your_offer, "Accepted!", True
            counter = min(self.limit, int(your_offer * 0.9))
            return counter, f"My counter: ₹{counter:,}", False

def test_auto_role_detection():
    """Test the automatic role detection capability"""

    test_products = [
        Product("Alphonso Mangoes", "Mangoes", 100, "A", "Ratnagiri", 180000, {"export_grade": True}),
        Product("Kesar Mangoes", "Mangoes", 150, "B", "Gujarat", 150000, {"export_grade": False})
    ]

    agent = YourDualRoleAgent("AutoNegotiator")

    print("="*60)
    print("TESTING AUTOMATIC ROLE DETECTION")
    print("="*60)

    # Test against both buyer and seller opponents
    for opponent_role in ["buyer", "seller"]:
        print(f"\n{'='*30} TESTING VS {opponent_role.upper()} {'='*30}")

        for product in test_products:
            if opponent_role == "seller":
                your_role = "buyer"
                your_limit = int(product.base_market_price * 1.1)  # Budget
                opponent_limit = int(product.base_market_price * 0.85)  # Min price
            else:
                your_role = "seller"
                your_limit = int(product.base_market_price * 0.9)  # Min price
                opponent_limit = int(product.base_market_price * 1.2)  # Budget

            print(f"\n{product.name} - vs {opponent_role}")
            print(f"My role: {your_role}, My limit: ₹{your_limit:,}")

            # Run test negotiation
            opponent = MockOpponentAgent(opponent_role, opponent_limit)
            context = NegotiationContext(
                product=product,
                your_role=your_role,
                your_limit=your_limit,
                current_round=0,
                opponent_offers=[],
                your_offers=[],
                messages=[]
            )

            # Opponent opens
            opp_price, opp_msg = opponent.get_opening_offer(product)
            context.opponent_offers.append(opp_price)
            context.messages.append({"role": opponent_role, "message": opp_msg})

            # Your agent responds
            context.current_round = 1
            status, your_price, your_msg = agent.respond_to_offer(context, opp_price, opp_msg)

            # Detect if role detection worked (note: some tests set your_role pre-emptively)
            detected_correctly = (context.your_role == your_role)
            status_symbol = "✅" if detected_correctly else "❌"

            print(f"{status_symbol} Detected role: {context.your_role}")
            print(f"Opening: {opp_msg}")
            print(f"Response: {your_msg}")

            if status == DealStatus.ACCEPTED:
                print(f"💰 QUICK DEAL at ₹{your_price:,}")
            else:
                print(f"↪ Counter: ₹{your_price:,}")

if __name__ == "__main__":
    test_auto_role_detection()
