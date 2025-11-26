use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Order {
    pub id: Uuid,
    pub prover_id: String,
    pub price: u64,
    pub capacity: u32,
}

#[derive(Debug, Default)]
pub struct OrderBook {
    pub orders: HashMap<Uuid, Order>,
}

impl OrderBook {
    pub fn new() -> Self { Self { orders: HashMap::new() } }

    pub fn add_order(&mut self, order: Order) {
        self.orders.insert(order.id, order);
    }

    pub fn best_order(&self) -> Option<&Order> {
        self.orders.values().min_by_key(|o| o.price)
    }
}
