use crate::order_book::OrderBook;
use uuid::Uuid;

pub struct AuctionEngine;

impl AuctionEngine {
    pub fn new() -> Self { Self }

    /// Pick best orders to satisfy capacity
    pub fn run_auction(book: &OrderBook, capacity_needed: u32) -> Vec<Uuid> {
        let mut orders: Vec<_> = book.orders.values().collect();
        orders.sort_by_key(|o| o.price);
        let mut picked = Vec::new();
        let mut cap = capacity_needed;
        for o in orders {
            if cap == 0 { break; }
            picked.push(o.id);
            cap = cap.saturating_sub(o.capacity);
        }
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order_book::OrderBook;
    use uuid::Uuid;

    #[test]
    fn test_auction_simple() {
        let mut book = OrderBook::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        book.add_order(crate::order_book::Order { id: id1, prover_id: "p1".to_string(), price: 100, capacity: 2 });
        book.add_order(crate::order_book::Order { id: id2, prover_id: "p2".to_string(), price: 90, capacity: 5 });

        let winners = AuctionEngine::run_auction(&book, 3);
        assert!(winners.contains(&id2));
    }
}
