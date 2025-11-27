/**
 * Home Screen
 * 
 * Dashboard showing wallet overview, privacy status, and recent activity
 */

import React from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useQuery } from '@tanstack/react-query';

// Theme
const theme = {
  colors: {
    primary: '#6366f1',
    background: '#0f172a',
    card: '#1e293b',
    cardAlt: '#334155',
    text: '#f8fafc',
    textSecondary: '#94a3b8',
    success: '#10b981',
    warning: '#f59e0b',
  },
};

// Privacy Level Component
const PrivacyIndicator: React.FC<{ level: number }> = ({ level }) => {
  const levels = ['Transparent', 'Pseudonymous', 'Confidential', 'Private', 'Anonymous', 'Sovereign'];
  const colors = ['#6b7280', '#3b82f6', '#8b5cf6', '#6366f1', '#4f46e5', '#10b981'];

  return (
    <View style={styles.privacyContainer}>
      <Text style={styles.privacyLabel}>Privacy Level</Text>
      <View style={styles.privacyLevelRow}>
        <View style={[styles.privacyDot, { backgroundColor: colors[level] }]} />
        <Text style={[styles.privacyLevel, { color: colors[level] }]}>
          {levels[level]}
        </Text>
      </View>
      <View style={styles.privacyBar}>
        {Array.from({ length: 6 }).map((_, i) => (
          <View
            key={i}
            style={[
              styles.privacySegment,
              { backgroundColor: i <= level ? colors[level] : theme.colors.cardAlt },
            ]}
          />
        ))}
      </View>
    </View>
  );
};

// Balance Card Component
const BalanceCard: React.FC = () => {
  return (
    <View style={styles.balanceCard}>
      <Text style={styles.balanceLabel}>Total Balance</Text>
      <Text style={styles.balanceAmount}>$24,567.89</Text>
      <View style={styles.balanceChange}>
        <Ionicons name="trending-up" size={16} color={theme.colors.success} />
        <Text style={styles.balanceChangeText}>+3.24% today</Text>
      </View>
    </View>
  );
};

// Quick Action Button
const QuickAction: React.FC<{
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
}> = ({ icon, label, onPress }) => (
  <TouchableOpacity style={styles.quickAction} onPress={onPress}>
    <View style={styles.quickActionIcon}>
      <Ionicons name={icon} size={24} color={theme.colors.primary} />
    </View>
    <Text style={styles.quickActionLabel}>{label}</Text>
  </TouchableOpacity>
);

// Recent Transaction Component
const RecentTransaction: React.FC<{
  type: 'send' | 'receive' | 'shield';
  amount: string;
  status: string;
  time: string;
}> = ({ type, amount, status, time }) => {
  const icons: Record<string, keyof typeof Ionicons.glyphMap> = {
    send: 'arrow-up-circle',
    receive: 'arrow-down-circle',
    shield: 'shield-checkmark',
  };

  const colors = {
    send: theme.colors.warning,
    receive: theme.colors.success,
    shield: theme.colors.primary,
  };

  return (
    <View style={styles.transaction}>
      <View style={[styles.transactionIcon, { backgroundColor: colors[type] + '20' }]}>
        <Ionicons name={icons[type]} size={24} color={colors[type]} />
      </View>
      <View style={styles.transactionDetails}>
        <Text style={styles.transactionType}>
          {type === 'send' ? 'Sent' : type === 'receive' ? 'Received' : 'Shielded'}
        </Text>
        <Text style={styles.transactionTime}>{time}</Text>
      </View>
      <View style={styles.transactionAmount}>
        <Text style={[styles.transactionValue, { color: type === 'send' ? theme.colors.warning : theme.colors.success }]}>
          {type === 'send' ? '-' : '+'}{amount}
        </Text>
        <Text style={styles.transactionStatus}>{status}</Text>
      </View>
    </View>
  );
};

export default function HomeScreen() {
  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1500);
  }, []);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.colors.primary}
          />
        }
      >
        {/* Balance Card */}
        <BalanceCard />

        {/* Privacy Level Indicator */}
        <PrivacyIndicator level={3} />

        {/* Quick Actions */}
        <View style={styles.quickActionsContainer}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.quickActionsRow}>
            <QuickAction icon="send-outline" label="Send" onPress={() => {}} />
            <QuickAction icon="download-outline" label="Receive" onPress={() => {}} />
            <QuickAction icon="shield-checkmark-outline" label="Shield" onPress={() => {}} />
            <QuickAction icon="swap-horizontal-outline" label="Bridge" onPress={() => {}} />
          </View>
        </View>

        {/* Recent Transactions */}
        <View style={styles.transactionsContainer}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <RecentTransaction
            type="shield"
            amount="1.5 ETH"
            status="Confirmed"
            time="2 hours ago"
          />
          <RecentTransaction
            type="send"
            amount="0.5 ETH"
            status="Confirmed"
            time="5 hours ago"
          />
          <RecentTransaction
            type="receive"
            amount="2.0 ETH"
            status="Confirmed"
            time="Yesterday"
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  scrollView: {
    flex: 1,
    padding: 16,
  },
  balanceCard: {
    backgroundColor: theme.colors.card,
    borderRadius: 20,
    padding: 24,
    marginBottom: 16,
  },
  balanceLabel: {
    color: theme.colors.textSecondary,
    fontSize: 14,
    marginBottom: 8,
  },
  balanceAmount: {
    color: theme.colors.text,
    fontSize: 36,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  balanceChange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  balanceChangeText: {
    color: theme.colors.success,
    fontSize: 14,
    marginLeft: 4,
  },
  privacyContainer: {
    backgroundColor: theme.colors.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  privacyLabel: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginBottom: 8,
  },
  privacyLevelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  privacyDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  privacyLevel: {
    fontSize: 18,
    fontWeight: '600',
  },
  privacyBar: {
    flexDirection: 'row',
    gap: 4,
  },
  privacySegment: {
    flex: 1,
    height: 4,
    borderRadius: 2,
  },
  sectionTitle: {
    color: theme.colors.text,
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  quickActionsContainer: {
    marginBottom: 24,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickAction: {
    alignItems: 'center',
  },
  quickActionIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: theme.colors.card,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  quickActionLabel: {
    color: theme.colors.textSecondary,
    fontSize: 12,
  },
  transactionsContainer: {
    marginBottom: 24,
  },
  transaction: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  transactionIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  transactionDetails: {
    flex: 1,
  },
  transactionType: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '500',
  },
  transactionTime: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
  transactionAmount: {
    alignItems: 'flex-end',
  },
  transactionValue: {
    fontSize: 16,
    fontWeight: '600',
  },
  transactionStatus: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
});
