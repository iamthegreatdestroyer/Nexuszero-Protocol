//! Proof Marketplace

pub mod order_book;
pub mod auction;

pub use order_book::*;
pub use auction::*;

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_order_book() {
        let mut book = OrderBook::new();
        let id = Uuid::new_v4();
        let order = Order { id, prover_id: "prover1".to_string(), price: 100, capacity: 5 };
        book.add_order(order.clone());
        assert_eq!(book.orders.len(), 1);

        let best = book.best_order().unwrap();
        assert_eq!(best.price, 100);
    }
}
