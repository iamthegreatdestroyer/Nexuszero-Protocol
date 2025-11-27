/**
 * Transact Screen
 * 
 * Create private transactions, shield assets, and manage transfers
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  Keyboard,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

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
    error: '#ef4444',
    border: '#475569',
  },
};

// Privacy levels
const privacyLevels = [
  { level: 0, name: 'Transparent', desc: 'Public blockchain parity', color: '#6b7280' },
  { level: 1, name: 'Pseudonymous', desc: 'Address obfuscation', color: '#3b82f6' },
  { level: 2, name: 'Confidential', desc: 'Encrypted amounts', color: '#8b5cf6' },
  { level: 3, name: 'Private', desc: 'Full transaction privacy', color: '#6366f1' },
  { level: 4, name: 'Anonymous', desc: 'Unlinkable transactions', color: '#4f46e5' },
  { level: 5, name: 'Sovereign', desc: 'Maximum privacy, ZK everything', color: '#10b981' },
];

// Transaction type selector
type TransactionType = 'send' | 'shield' | 'unshield' | 'bridge';

const TransactionTypeButton: React.FC<{
  type: TransactionType;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  active: boolean;
  onPress: () => void;
}> = ({ type, label, icon, active, onPress }) => (
  <TouchableOpacity
    style={[styles.typeButton, active && styles.typeButtonActive]}
    onPress={onPress}
  >
    <Ionicons
      name={icon}
      size={20}
      color={active ? theme.colors.text : theme.colors.textSecondary}
    />
    <Text style={[styles.typeButtonText, active && styles.typeButtonTextActive]}>
      {label}
    </Text>
  </TouchableOpacity>
);

// Privacy Level Selector
const PrivacyLevelSelector: React.FC<{
  selectedLevel: number;
  onSelect: (level: number) => void;
}> = ({ selectedLevel, onSelect }) => (
  <View style={styles.privacySelector}>
    <Text style={styles.inputLabel}>Privacy Level</Text>
    <View style={styles.privacyLevels}>
      {privacyLevels.map(({ level, name, desc, color }) => (
        <TouchableOpacity
          key={level}
          style={[
            styles.privacyLevel,
            selectedLevel === level && { borderColor: color, backgroundColor: color + '10' },
          ]}
          onPress={() => {
            Haptics.selectionAsync();
            onSelect(level);
          }}
        >
          <View style={[styles.privacyDot, { backgroundColor: color }]} />
          <View style={styles.privacyInfo}>
            <Text style={styles.privacyName}>{name}</Text>
            <Text style={styles.privacyDesc}>{desc}</Text>
          </View>
          {selectedLevel === level && (
            <Ionicons name="checkmark-circle" size={20} color={color} />
          )}
        </TouchableOpacity>
      ))}
    </View>
  </View>
);

export default function TransactScreen() {
  const [txType, setTxType] = useState<TransactionType>('send');
  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [privacyLevel, setPrivacyLevel] = useState(3);
  const [isLoading, setIsLoading] = useState(false);

  const handleTransaction = async () => {
    Keyboard.dismiss();

    if (!amount || parseFloat(amount) <= 0) {
      Alert.alert('Invalid Amount', 'Please enter a valid amount');
      return;
    }

    if ((txType === 'send' || txType === 'bridge') && !recipient) {
      Alert.alert('Missing Recipient', 'Please enter a recipient address');
      return;
    }

    setIsLoading(true);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    // Simulate transaction
    setTimeout(() => {
      setIsLoading(false);
      Alert.alert(
        'Transaction Initiated',
        `Your ${txType} transaction with privacy level ${privacyLevels[privacyLevel].name} is being processed.`,
        [
          {
            text: 'View Status',
            onPress: () => {},
          },
          { text: 'OK' },
        ]
      );
    }, 2000);
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView} keyboardShouldPersistTaps="handled">
        {/* Transaction Type Selector */}
        <View style={styles.typeSelector}>
          <TransactionTypeButton
            type="send"
            label="Send"
            icon="send-outline"
            active={txType === 'send'}
            onPress={() => setTxType('send')}
          />
          <TransactionTypeButton
            type="shield"
            label="Shield"
            icon="shield-checkmark-outline"
            active={txType === 'shield'}
            onPress={() => setTxType('shield')}
          />
          <TransactionTypeButton
            type="unshield"
            label="Unshield"
            icon="shield-outline"
            active={txType === 'unshield'}
            onPress={() => setTxType('unshield')}
          />
          <TransactionTypeButton
            type="bridge"
            label="Bridge"
            icon="swap-horizontal-outline"
            active={txType === 'bridge'}
            onPress={() => setTxType('bridge')}
          />
        </View>

        {/* Amount Input */}
        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Amount</Text>
          <View style={styles.amountInputWrapper}>
            <TextInput
              style={styles.amountInput}
              placeholder="0.00"
              placeholderTextColor={theme.colors.textSecondary}
              value={amount}
              onChangeText={setAmount}
              keyboardType="decimal-pad"
            />
            <TouchableOpacity style={styles.tokenSelector}>
              <Text style={styles.tokenSelectorText}>ETH</Text>
              <Ionicons name="chevron-down" size={16} color={theme.colors.textSecondary} />
            </TouchableOpacity>
          </View>
          <View style={styles.balanceRow}>
            <Text style={styles.balanceText}>Available: 2.5 ETH</Text>
            <TouchableOpacity onPress={() => setAmount('2.5')}>
              <Text style={styles.maxButton}>MAX</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Recipient Input (for send/bridge) */}
        {(txType === 'send' || txType === 'bridge') && (
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>
              {txType === 'bridge' ? 'Destination Address' : 'Recipient Address'}
            </Text>
            <View style={styles.addressInputWrapper}>
              <TextInput
                style={styles.addressInput}
                placeholder="0x..."
                placeholderTextColor={theme.colors.textSecondary}
                value={recipient}
                onChangeText={setRecipient}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TouchableOpacity style={styles.scanButton}>
                <Ionicons name="scan-outline" size={20} color={theme.colors.primary} />
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Bridge Chain Selector */}
        {txType === 'bridge' && (
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>Bridge To</Text>
            <View style={styles.chainSelector}>
              {['Polygon', 'BSC', 'Arbitrum', 'Optimism'].map((chain) => (
                <TouchableOpacity key={chain} style={styles.chainOption}>
                  <Text style={styles.chainText}>{chain}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Privacy Level Selector */}
        <PrivacyLevelSelector
          selectedLevel={privacyLevel}
          onSelect={setPrivacyLevel}
        />

        {/* Transaction Summary */}
        <View style={styles.summary}>
          <Text style={styles.summaryTitle}>Transaction Summary</Text>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Type</Text>
            <Text style={styles.summaryValue}>
              {txType.charAt(0).toUpperCase() + txType.slice(1)}
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Amount</Text>
            <Text style={styles.summaryValue}>{amount || '0'} ETH</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Privacy</Text>
            <Text style={[styles.summaryValue, { color: privacyLevels[privacyLevel].color }]}>
              {privacyLevels[privacyLevel].name}
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Est. Proof Time</Text>
            <Text style={styles.summaryValue}>
              ~{[0, 50, 100, 250, 500, 1000][privacyLevel]}ms
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Est. Gas</Text>
            <Text style={styles.summaryValue}>
              ~{[21000, 50000, 100000, 200000, 350000, 500000][privacyLevel].toLocaleString()}
            </Text>
          </View>
        </View>

        {/* Submit Button */}
        <TouchableOpacity
          style={[styles.submitButton, isLoading && styles.submitButtonDisabled]}
          onPress={handleTransaction}
          disabled={isLoading}
        >
          {isLoading ? (
            <Text style={styles.submitButtonText}>Processing...</Text>
          ) : (
            <>
              <Ionicons name="shield-checkmark" size={20} color={theme.colors.text} />
              <Text style={styles.submitButtonText}>
                Confirm {txType.charAt(0).toUpperCase() + txType.slice(1)}
              </Text>
            </>
          )}
        </TouchableOpacity>
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
  typeSelector: {
    flexDirection: 'row',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 4,
    marginBottom: 24,
  },
  typeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
  typeButtonActive: {
    backgroundColor: theme.colors.primary,
  },
  typeButtonText: {
    color: theme.colors.textSecondary,
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 6,
  },
  typeButtonTextActive: {
    color: theme.colors.text,
  },
  inputContainer: {
    marginBottom: 24,
  },
  inputLabel: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  amountInputWrapper: {
    flexDirection: 'row',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    overflow: 'hidden',
  },
  amountInput: {
    flex: 1,
    color: theme.colors.text,
    fontSize: 24,
    fontWeight: '600',
    padding: 16,
  },
  tokenSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    backgroundColor: theme.colors.cardAlt,
  },
  tokenSelectorText: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginRight: 4,
  },
  balanceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  balanceText: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
  maxButton: {
    color: theme.colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  addressInputWrapper: {
    flexDirection: 'row',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    overflow: 'hidden',
  },
  addressInput: {
    flex: 1,
    color: theme.colors.text,
    fontSize: 16,
    padding: 16,
  },
  scanButton: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  chainSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chainOption: {
    backgroundColor: theme.colors.card,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
  },
  chainText: {
    color: theme.colors.text,
    fontSize: 14,
  },
  privacySelector: {
    marginBottom: 24,
  },
  privacyLevels: {
    gap: 8,
  },
  privacyLevel: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  privacyDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 12,
  },
  privacyInfo: {
    flex: 1,
  },
  privacyName: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
  },
  privacyDesc: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
  summary: {
    backgroundColor: theme.colors.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 24,
  },
  summaryTitle: {
    color: theme.colors.text,
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  summaryLabel: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
  summaryValue: {
    color: theme.colors.text,
    fontSize: 14,
    fontWeight: '500',
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primary,
    borderRadius: 12,
    padding: 18,
    marginBottom: 32,
  },
  submitButtonDisabled: {
    opacity: 0.7,
  },
  submitButtonText: {
    color: theme.colors.text,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
});
